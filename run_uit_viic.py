from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, LearningRateMonitor

from transformers.optimization import get_linear_schedule_with_warmup

from src.data import ImageTextPair, ImageCaptioningCollator
from src.tokenization import BunTokenizer
from src.vision_language import ImageTextConfig, ImageTextForCausalLM

import torch_xla.core.xla_model as xm


class Wrapper(pl.LightningModule):
    def __init__(self, model_config,
                 warmup_ratio: float,
                 learn_rate: float,
                 use_pretrain: bool) -> None:
        super().__init__()
        self.warmup_ratio = warmup_ratio
        self.learn_rate = learn_rate
        self.save_hyperparameters()

        self.model = ImageTextForCausalLM(model_config)
        if use_pretrain:
            self.model.load_state_dict(torch.load('lightning_logs/5/checkpoints/imagetext.pt'), strict=False)

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
        return [opt], [get_linear_schedule_with_warmup(opt,
                                                       self.trainer.estimated_stepping_batches * self.warmup_ratio,
                                                       self.trainer.estimated_stepping_batches)]


if '__main__' == __name__:
    seed_everything(5)
    train_ic = ImageTextPair(Path('/home/dinhanhx/data/'),
                             Path('/home/dinhanhx/data/TranslateCOCO2017/'),
                             split='train_uit_viic')
    val_ic = ImageTextPair(Path('/home/dinhanhx/data/'),
                           Path('/home/dinhanhx/data/TranslateCOCO2017/'),
                           split='val_uit_viic')
    train_val_ic = ConcatDataset([train_ic, val_ic])

    bun_tokenizer = BunTokenizer.from_pretrained('vinai/bartpho-syllable')
    config = ImageTextConfig.from_json_file('assets/imagetext-casual-base-config.json')

    ic_collator = ImageCaptioningCollator(bun_tokenizer,
                                          image_size=config.image_size,
                                          patch_size=config.patch_size)

    sampler = DistributedSampler(
            train_val_ic, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
        )
    dataloader = DataLoader(train_val_ic,
                            batch_size=8,
                            sampler=sampler,
                            num_workers=24,
                            collate_fn=ic_collator,
                            drop_last=True)

    wrapper = Wrapper(config,
                      warmup_ratio=0.2,
                      learn_rate=5.0e-05,
                      use_pretrain=True)

    trainer = Trainer(logger=TensorBoardLogger("uit_viic_logs"),
                      max_epochs=8,
                      log_every_n_steps=100,
                      accelerator='tpu', devices=8,
                      callbacks=[RichProgressBar(),
                                 ModelCheckpoint(every_n_train_steps=100),
                                 LearningRateMonitor(logging_interval='step')],
                      strategy="tpu_spawn_debug",
                      precision='bf16',
                      profiler='xla')
    trainer.fit(wrapper, dataloader)
