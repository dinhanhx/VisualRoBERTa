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
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
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

    def forward(self, batch):
        return self.model(**batch)

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
        # return {"loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.learn_rate)
        return [opt], [
            get_linear_schedule_with_warmup(
                opt,
                self.trainer.estimated_stepping_batches * self.warmup_ratio,
                self.trainer.estimated_stepping_batches,
            )
        ]


if "__main__" == __name__:
    seed_everything(5)
    pretrain_task = PretrainTask(
        Path("/home/dinhanhx/data"),
        Path("/home/dinhanhx/data/TranslateCOCO2017"),
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
        sampler=sampler,
        num_workers=24,
        collate_fn=pretrain_collator,
        drop_last=True,
    )

    wrapper = Wrapper(config, warmup_ratio=0.2, learn_rate=5.0e-05, use_phobert=False)

    trainer = Trainer(
        logger=[CSVLogger("logs"), TensorBoardLogger("logs")],
        max_epochs=2,
        accelerator="tpu",
        devices=8,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(every_n_train_steps=1000),
            LearningRateMonitor(logging_interval="step"),
        ],
        strategy="tpu_spawn_debug",
        precision="bf16",
        profiler="xla",
    )
    trainer.fit(wrapper, dataloader)
