from pathlib import Path

import pytorch_lightning as pl
import torch
import torch_xla.core.xla_model as xm
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_linear_schedule_with_warmup

from src.data import PretrainCollator, PretrainTask
from src.tokenization import BunTokenizer
from src.vision_language import ImageTextConfig, ImageTextForPretraining


class Wrapper(pl.LightningModule):
    def __init__(
        self, model_config, warmup_ratio: float, learn_rate: float, use_phobert: bool
    ) -> None:
        super().__init__()
        self.warmup_ratio = warmup_ratio
        self.learn_rate = learn_rate
        self.save_hyperparameters()

        self.model = ImageTextForPretraining(model_config)  # type: ignore
        if use_phobert:
            self.model.imagetext.econder.load_state_dict(
                torch.load("assets/phobert-base-encoder.pt")
            )
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        loss = self.model(**batch).loss
        self.log("train_loss", loss)
        self.manual_backward(loss)

        opt.step()
        sch = self.lr_schedulers()
        sch.step()

        xm.mark_step()
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.learn_rate)
        opt_list = [opt]

        calculated_warmup_steps = (
            self.trainer.estimated_stepping_batches * self.warmup_ratio
        )
        lrs = {
            "scheduler": get_linear_schedule_with_warmup(
                opt, calculated_warmup_steps, self.trainer.estimated_stepping_batches
            ),
            "interval": "step",
            "frequency": 1,
        }
        lrs_list = [lrs]
        return opt_list, lrs_list


if "__main__" == __name__:
    seed_everything(5)
    pretrain_task = PretrainTask(
        Path("/home/anhvd_m21ict/data/coco-2017-images"),
        Path("/home/anhvd_m21ict/data/coco-2017-vi/vi"),
        split="train",
    )

    bun_tokenizer = BunTokenizer.from_pretrained("vinai/bartpho-syllable")
    config = ImageTextConfig.from_json_file("assets/imagetext-base-config.json")

    pretrain_collator = PretrainCollator(
        bun_tokenizer, image_size=config.image_size, patch_size=config.patch_size
    )
    sampler = DistributedSampler(
        pretrain_task,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )
    dataloader = DataLoader(
        pretrain_task,
        batch_size=8,
        num_workers=24,
        collate_fn=pretrain_collator,
        drop_last=True,
        sampler=sampler
    )

    wrapper = Wrapper(config, warmup_ratio=0.2, learn_rate=5.0e-05, use_phobert=False)

    do_every_n_steps = 1000
    root_dir = "pls-logs"

    trainer = Trainer(
        enable_checkpointing=True,
        default_root_dir=root_dir,
        logger=[TensorBoardLogger(root_dir)],
        max_epochs=2,
        accelerator="tpu",
        devices=8,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(every_n_train_steps=do_every_n_steps),
            LearningRateMonitor(logging_interval="step"),
        ],
        precision="bf16-mixed",
        log_every_n_steps=do_every_n_steps,
        enable_model_summary=False,
    )
    trainer.fit(wrapper, dataloader)
