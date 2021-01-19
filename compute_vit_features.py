import os
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
import pytorch_lightning as pl

from utils.utils import init_params, seed_reproducer, mkdir
from settings import classes, n_classes
from train_vit import SystemViT
from train_cam import System
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":

    # Make experiment reproducible
    seed_reproducer(2020)

    hparams = init_params()

    mkdir(hparams.vit_features_dir)
    if hparams.compute_vit_features:
        model = SystemViT(hparams, n_classes)
    else:
        model = System(hparams, n_classes)
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        min_epochs=10,
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=0,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=True,
        weights_summary=None,
        gradient_clip_val=hparams.gradient_clip_val
    )
    model.load_state_dict(torch.load(hparams.load_model)["state_dict"])

    model.to("cuda")
    model.eval()

    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for data in [hparams.data_train, hparams.data_val]:
        val_data = pd.read_csv(data)

        for value in tqdm(val_data.values):
            # Load data
            filename = value[0]
            img_pil = Image.open(filename).convert('RGB')
            label = value[1:].astype(int)
            label = torch.Tensor(label)

            # Forward to get kd vectors
            x = transform_data(img_pil)
            x = x.to("cuda")
            features = model.forward_features(x.unsqueeze(0))

            # save kd vectors
            filename = os.path.join(hparams.vit_features_dir, os.path.splitext(os.path.basename(filename))[0] + '.npy')
            np.save(filename,
                    features.detach().cpu())
