import os
import pandas as pd

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn as nn

from sklearn.metrics import accuracy_score
from loss_scheduler_optimizer import WarmupLinearSchedule, WarmupCosineSchedule

from utils.utils import init_params, seed_reproducer, init_logger, mkdir
from settings import classes, n_classes
from dataset import generate_dataloaders, generate_transforms

import timm

class SystemViT(pl.LightningModule):

    def __init__(self, hparams, n_classes):
        super(SystemViT, self).__init__()

        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(768, n_classes)

        self.my_logger = init_logger("terminal", hparams.log_dir)

        self.lr = hparams.lr
        self.wt_dec = hparams.wt_dec
        self.criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat  = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {'val_loss': loss,
                'y_hat': y_hat,
                'y': y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y_hat_all = torch.round(torch.sigmoid(torch.cat([output["y_hat"] for output in outputs]))).cpu()
        y_all = torch.round(torch.cat([output["y"] for output in outputs]).cpu())
        val_accuracy = accuracy_score(y_hat_all, y_all)

        # terminal logs
        self.my_logger.info(
            f"{self.current_epoch}  / {hparams.max_epochs}| "
            f"val_loss : {avg_loss:.4f} | "
            f"val_accuracy : {val_accuracy:.4f}"
        )

        return {'val_loss': avg_loss,
                'val_accuracy': val_accuracy}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.multilabel_soft_margin_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([
            {"params": self.backbone.parameters(), "lr": 1e-1},
            {"params": self.classifier.parameters(), "lr": 1e-1}
        ],
            lr=0.1,
            momentum=0.9
            )

        warmup_steps = 500
        t_total = 1000

        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
        return [optimizer], [scheduler]

if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)
    hparams = init_params()

    data_train = pd.read_csv(hparams.data_train)
    data_val = pd.read_csv(hparams.data_val)

    transforms = generate_transforms([224, 224])
    train_dataloader, val_dataloader = generate_dataloaders(hparams, data_train, data_val, transforms)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=1,
        mode="max",
        filepath=os.path.join(hparams.log_dir, "best_model"),
    )
    early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=10, mode="max", verbose=True)

    system = SystemViT(hparams, n_classes)

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        min_epochs=10,
        max_epochs=hparams.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        progress_bar_refresh_rate=0,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=True,
        weights_summary=None,
        gradient_clip_val=hparams.gradient_clip_val
    )

    trainer.fit(system, train_dataloader, val_dataloader)