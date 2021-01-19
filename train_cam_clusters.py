import os
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score

from utils.utils import init_params, seed_reproducer, init_logger
from loss_scheduler_optimizer import PolyOptimizer
from settings import classes, n_classes
from dataset import generate_dataloaders, generate_transforms

from network.resnet38_cls import Net
import network

class System(pl.LightningModule):

    def __init__(self, hparams, n_classes):
        super(System, self).__init__()

        self.model = Net()

        self.subcategory_classifier = nn.Conv2d(4096, hparams.n_clusters, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.subcategory_classifier.weight)

        self.my_logger = init_logger("terminal", hparams.log_dir)

        self.criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

    def forward(self, x):
        x = self.model.conv1a(x)

        x = self.model.b2(x)
        x = self.model.b2_1(x)
        x = self.model.b2_2(x)

        x = self.model.b3(x)
        x = self.model.b3_1(x)
        x = self.model.b3_2(x)

        x = self.model.b4(x)
        x = self.model.b4_1(x)
        x = self.model.b4_2(x)
        x = self.model.b4_3(x)
        x = self.model.b4_4(x)
        x = self.model.b4_5(x)

        x, conv4 = self.model.b5(x, get_x_bn_relu=True)
        x = self.model.b5_1(x)
        x = self.model.b5_2(x)

        x, conv5 = self.model.b6(x, get_x_bn_relu=True)

        x = self.model.b7(x)
        x = F.relu(self.model.bn7(x))

        # Forward towards classification labels
        x = self.model.dropout7(x)
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x_cls = self.model.fc8(x)
        x_cls = x_cls.view(x_cls.size(0), -1)

        x_cls2 = self.subcategory_classifier(x)
        x_cls2 = x_cls2.view(x_cls2.size(0), -1)

        return x_cls, x_cls2

    def forward_cam(self, x):
        x = self.model.forward_cam(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, y_clusters = batch

        y_hat, y_hat2 = self.forward(x)

        loss = self.criterion(y_hat, y)
        loss2 = self.criterion(y_hat2, y_clusters)
        loss = loss + hparams.knowledge_distillation_loss_weight * loss2
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y, y_clusters = batch
        y_hat, y_hat2 = self.forward(x)
        loss = F.multilabel_soft_margin_loss(y_hat, y)
        loss2 = F.multilabel_soft_margin_loss(y_hat2, y_clusters)
        return {'val_loss': loss,
                'y_hat': y_hat,
                'y': y,
                'val_loss_clusters': loss2,
                'y_clusters_hat': y_hat2,
                'y_clusters': y_clusters}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss2 = torch.stack([x['val_loss_clusters'] for x in outputs]).mean()

        y_clusters_hat_all = torch.round(F.sigmoid(torch.cat([output["y_clusters_hat"] for output in outputs]))).cpu()
        y_clusters_all = torch.round(torch.cat([output["y_clusters"] for output in outputs]).cpu())
        val_cluster_accuracy = accuracy_score(y_clusters_hat_all, y_clusters_all)

        y_hat_all = torch.round(F.sigmoid(torch.cat([output["y_hat"] for output in outputs]))).cpu()
        y_all = torch.round(torch.cat([output["y"] for output in outputs]).cpu())
        val_accuracy = accuracy_score(y_hat_all, y_all)

        # terminal logs
        self.my_logger.info(
            f"{self.current_epoch}  / {hparams.max_epochs}| "
            f"val_loss : {avg_loss:.4f} | "
            f"val_accuracy : {val_accuracy:.4f}"
            f"val_cluster_loss : {avg_loss2:.4f} | "
            f"val_cluster_accuracy : {val_cluster_accuracy:.4f}"
        )

        return {'val_loss': avg_loss,
                'val_accuracy': val_accuracy}

    def test_step(self, batch, batch_nb):
        x, y, y_clusters = batch
        y_hat, _ = self.forward(x)
        return {'test_loss': F.multilabel_soft_margin_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'progress_bar': logs}

    def configure_optimizers(self):
        param_groups = self.model.get_parameter_groups()

        param_groups_v2 = self.subcategory_classifier.parameters()
        optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': hparams.lr, 'weight_decay': hparams.wt_dec},
            {'params': param_groups[1], 'lr': hparams.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': hparams.lr, 'weight_decay': hparams.wt_dec},
            {'params': param_groups[3], 'lr': hparams.lr, 'weight_decay': 0},
            {'params': param_groups_v2, 'lr': hparams.lr, 'weight_decay': 0},
        ], lr=hparams.lr, weight_decay=hparams.wt_dec, max_step=100000)
        return optimizer


if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)
    hparams = init_params()

    # Data
    data_train = pd.read_csv(hparams.data_train)
    data_val = pd.read_csv(hparams.data_val)

    transforms = generate_transforms(hparams.image_size)
    train_dataloader, val_dataloader = generate_dataloaders(hparams, data_train, data_val, transforms, knowledge_distillation=hparams.knowledge_distillation)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=5,
        mode="max",
        filepath=os.path.join(hparams.log_dir, "{epoch}-{val_accuracy:.4f}"),
    )
    early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=10, mode="max", verbose=True)

    # From pretrained weights
    if len(hparams.load_model) == 0:
        system = System(hparams, n_classes)
        weights_dict = network.resnet38d.convert_mxnet_to_torch(hparams.pretrained_weights)
        system.model.load_state_dict(weights_dict, strict=False)
    # from checkpoints
    else:
        system = System(hparams, n_classes)
        system.load_state_dict(torch.load(hparams.load_model)["state_dict"])
        
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        min_epochs=1,
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