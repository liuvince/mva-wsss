import os
import pandas as pd
import random
import sys

import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score

from utils.utils import init_params, seed_reproducer, init_logger
from settings import classes, n_classes
from dataset import generate_dataloaders, generate_transforms
from loss_scheduler_optimizer import PolyOptimizer, mixup_data, mixup_criterion, mixup_data_kd, entropy_regularization_loss, concentration_loss

from network.resnet38_cls import Net
import network

from FMix.implementations.lightning import FMix

class System(pl.LightningModule):

    def __init__(self, hparams, n_classes):
        super(System, self).__init__()

        self.model = Net()
        self.my_logger = init_logger("terminal", hparams.log_dir)

        self.criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')

        self.fmix = FMix(size=(300, 300))
    def forward(self, x):
        x_cls = self.model.forward(x)
        return x_cls

    def forward_label_cam_manifold_mixup(self, x, target, mixup_hidden = True,  mixup_alpha = 0.2, layer_mix=None):

        if mixup_hidden == True:
            if layer_mix == None:
                layer_mix = random.randint(2, 7)

            out = x

            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.model.conv1a(out)

            out = self.model.b2(out)
            out = self.model.b2_1(out)
            out = self.model.b2_2(out)

            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.model.b3(out)
            out = self.model.b3_1(out)
            out = self.model.b3_2(out)

            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.model.b4(out)
            out = self.model.b4_1(out)
            out = self.model.b4_2(out)
            out = self.model.b4_3(out)
            out = self.model.b4_4(out)
            out = self.model.b4_5(out)

            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out, conv4 = self.model.b5(out, get_x_bn_relu=True)
            out = self.model.b5_1(out)
            out = self.model.b5_2(out)

            if layer_mix == 6:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out, conv5 = self.model.b6(out, get_x_bn_relu=True)

            if layer_mix == 7:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.model.b7(out)
            out = F.relu(self.model.bn7(out))

            # Forward towards cam generation
            x_cam = F.conv2d(out, self.model.fc8.weight)
            x_cam = F.relu(x_cam)

            # Forward towards classification labels
            out = self.model.dropout7(out)
            out = F.avg_pool2d(
                out, kernel_size=(out.size(2), out.size(3)), padding=0)

            x_cls = self.model.fc8(out)
            x_cls = x_cls.view(x_cls.size(0), -1)

            lam = torch.tensor(lam).cuda()
            #lam = lam.repeat(y_a.size())

            return x_cls, y_a, y_b, lam, x_cam

    def forward_label_cam(self, x):
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

        # Forward towards cam generation
        x_cam = F.conv2d(x, self.model.fc8.weight)
        x_cam = F.relu(x_cam)

        # Forward towards classification labels
        x = self.model.dropout7(x)
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x_cls = self.model.fc8(x)
        x_cls = x_cls.view(x_cls.size(0), -1)

        return x_cls, x_cam

    def forward_cam(self, x):
        x = self.model.forward_cam(x)
        return x

    def forward_features(self, x):
        x = self.model.forward_super(x)
        x = self.model.dropout7(x)
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        return x

    def training_step(self, batch, batch_nb):

        x, y = batch
        if hparams.mixup:
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            y_hat, x_cam = self.forward_label_cam(mixed_x)
            loss = mixup_criterion(self.criterion, y_hat, y_a, y_b, lam)
        elif hparams.manifold_mixup:
            y_hat, y_a, y_b, lam, x_cam = self.forward_label_cam_manifold_mixup(x, y)
            loss = mixup_criterion(self.criterion, y_hat, y_a, y_b, lam)
        elif hparams.fmix:
            x = self.fmix(x)
            y_hat, x_cam = self.forward_label_cam(x)
            y_a, y_b, lam = y, y[self.fmix.index], self.fmix.lam
            loss = self.criterion(y_hat, y_a) * lam + self.criterion(y_hat, y_b) * (1 - lam)
        else:
            y_hat, x_cam = self.forward_label_cam(x)
            loss = self.criterion(y_hat, y)

        if hparams.ent:
            if hparams.mixup or hparams.manifold_mixup or hparams.fmix:
                loss_a = lam * entropy_regularization_loss(y_a) + (1 - lam) * entropy_regularization_loss(y_b)
                loss += hparams.ent_loss_weight * loss_a
            else:
                loss += hparams.ent_loss_weight * entropy_regularization_loss(y_hat)
        if hparams.concent:
            x_cam = torch.softmax(x_cam, axis=1)
            if hparams.mixup or hparams.manifold_mixup or hparams.fmix:
                loss_b = lam *  concentration_loss(y_a, x_cam) + (1 - lam) *  concentration_loss(y_b, x_cam)
                loss += hparams.concent_loss_weight * loss_b
            else:
                loss += hparams.concent_loss_weight * concentration_loss(y, x_cam)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        x, y = batch

        y_hat, x_cam = self.forward_label_cam(x)
        loss = self.criterion(y_hat, y)

        return {'val_loss': loss,
                'y_hat': y_hat,
                'y': y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y_hat_all = torch.round(F.sigmoid(torch.cat([output["y_hat"] for output in outputs]))).cpu()
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
        param_groups = self.model.get_parameter_groups()
        optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': hparams.lr, 'weight_decay': hparams.wt_dec},
            {'params': param_groups[1], 'lr': hparams.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': hparams.lr, 'weight_decay': hparams.wt_dec},
            {'params': param_groups[3], 'lr': hparams.lr, 'weight_decay': 0}
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
        save_top_k=1,
        mode="max",
       # filepath=os.path.join(hparams.log_dir, "{epoch}-{val_accuracy:.4f}"),
        filepath=os.path.join(hparams.log_dir, "best_model"),

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

