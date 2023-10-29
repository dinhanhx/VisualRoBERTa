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
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_linear_schedule_with_warmup

from src.data import ImageCaptioningCollator, ImageTextPair
from src.tokenization import BunTokenizer
from src.vision_language import ImageTextConfig, ImageTextForCausalLM


class Wrapper(pl.LightningModule):
    def __init__(
        self,
        model_config,
        warmup_ratio: float,
        learn_rate: float,
        use_pretrain: bool,
        pretrain_model_path_str: str,
    ) -> None:
        super().__init__()
        self.warmup_ratio = warmup_ratio
        self.learn_rate = learn_rate
        self.pretrain_model_path_str = pretrain_model_path_str
        self.save_hyperparameters()

        self.model = ImageTextForCausalLM(model_config)
        if use_pretrain:
            self.model.load_state_dict(
                torch.load(self.pretrain_model_path_str), strict=False
            )
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        loss = self.model(**batch).loss
        self.log("train_loss", loss)

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
    train_ic = ImageTextPair(
        Path("/home/anhvd_m21ict/data/coco-2017-images"),
        Path("/home/anhvd_m21ict/data/coco-2017-vi/vi"),
        split="train_uit_viic",
    )
    val_ic = ImageTextPair(
        Path("/home/anhvd_m21ict/data/coco-2017-images"),
        Path("/home/anhvd_m21ict/data/coco-2017-vi/vi"),
        split="val_uit_viic",
    )
    train_val_ic = ConcatDataset([train_ic, val_ic])

    bun_tokenizer = BunTokenizer.from_pretrained("vinai/bartpho-syllable")
    config = ImageTextConfig.from_json_file("assets/imagetext-casual-base-config.json")

    ic_collator = ImageCaptioningCollator(
        bun_tokenizer, image_size=config.image_size, patch_size=config.patch_size
    )
    sampler = DistributedSampler(
        train_val_ic,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )
    dataloader = DataLoader(
        train_val_ic,
        batch_size=8,
        num_workers=24,
        collate_fn=ic_collator,
        drop_last=True,
        sampler=sampler
    )

    pretrain_model_path_str = (
        "pls-logs/lightning_logs/version_4/checkpoints/imagetext-base.pt"
    )

    wrapper = Wrapper(
        config,
        warmup_ratio=0.2,
        learn_rate=5.0e-05,
        use_pretrain=True,
        pretrain_model_path_str=pretrain_model_path_str,
    )

    do_every_n_steps = 100
    root_dir = "pls-uit-viic-logs"

    trainer = Trainer(
        enable_checkpointing=True,
        default_root_dir=root_dir,
        logger=[TensorBoardLogger(root_dir)],
        max_epochs=8,
        accelerator="tpu",
        devices=8,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                every_n_train_steps=do_every_n_steps
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        precision="bf16-mixed",
        log_every_n_steps=do_every_n_steps,
        enable_model_summary=False,
    )
    trainer.fit(wrapper, dataloader)
