import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl

from utils.utils import init_params, seed_reproducer, mkdir
import utils.imutils as imutils
from settings import classes, n_classes

if __name__ == "__main__":

    # Make experiment reproducible
    seed_reproducer(2020)

    hparams = init_params()

    # Model
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        min_epochs=10,
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=0,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=True,
        weights_summary=None,
        #use_dp=True,
        gradient_clip_val=hparams.gradient_clip_val
    )

    if hparams.knowledge_distillation:
        from train_cam_clusters import System
    else:
        from train_cam import System
    model = System(hparams, n_classes)
    model.load_state_dict(torch.load(hparams.load_model)["state_dict"])
    model.to("cuda")
    model.eval()
    torch.save(model.model.state_dict(), "model")