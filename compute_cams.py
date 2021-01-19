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

# Generation of CAMs inspired by: https://github.com/jiwoon-ahn/irn/blob/master/net/resnet50_cam.py

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

if __name__ == "__main__":

    # Make experiment reproducible
    seed_reproducer(2020)

    hparams = init_params()

    # Create directory of where the class activation maps are generated
    mkdir(hparams.cam_dir)

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

    # Data of which we want to compute the CAMs
    data = pd.read_csv(hparams.data_cam_generation)

    transform_data = transforms.Compose([
        np.asarray,
        imutils.normalize(),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])

    transform_flip_data = transforms.Compose([
        np.asarray,
        imutils.HorizontalFlip(),
        imutils.normalize(),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])

    # Iterate over each image
    for value in tqdm(data.values):
        # Path and label of the images
        filename = value[0]
        label = torch.Tensor(value[1:].astype(int))

        # Open the image
        img_pil = Image.open(filename).convert('RGB')
        width, height = img_pil.size

        size = (height, width)
        strided_size = get_strided_size(size, 4)
        strided_up_size = get_strided_up_size(size, 16)

        # Positive labels
        valid_cat = torch.nonzero(label)[:, 0]

        highres_cam_list = []
        strided_cam_list = []
        # TTA
        #for scale in [1, 0.5, 0.8, 1.2]:
        for scale in [1, 0.5, 1.5, 2]:
            # No FLIP
            img = img_pil.resize((int(width * scale) , int(height * scale)), Image.ANTIALIAS)

            x = transform_data(img)
            x = x.to("cuda")
            with torch.no_grad():
                outputs = model.forward_cam(x.unsqueeze(0))

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                     mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            strided_cam = strided_cam[valid_cat]
            highres_cam = highres_cam[valid_cat]

            strided_cam_list.append(strided_cam.detach().cpu().numpy())
            highres_cam_list.append(highres_cam.detach().cpu().numpy())
            del x
            torch.cuda.empty_cache()

            # FLIP
            x = transform_flip_data(img)
            x = x.to("cuda")
            with torch.no_grad():
                outputs = model.forward_cam(x.unsqueeze(0))

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            strided_cam = strided_cam[valid_cat]
            highres_cam = highres_cam[valid_cat]

            strided_cam_list.append(np.flip(strided_cam.detach().cpu().numpy(), axis=-1))
            highres_cam_list.append(np.flip(highres_cam.detach().cpu().numpy(), axis=-1))
            del x
            torch.cuda.empty_cache()

        # Average the obtained CAMs
        strided_cam = np.sum(strided_cam_list, axis=0)
        highres_cam = np.sum(highres_cam_list, axis=0)

        strided_cam = strided_cam / (np.max(strided_cam, (1, 2), keepdims=True) + 1e-5)
        highres_cam = highres_cam / (np.max(highres_cam, (1, 2), keepdims=True) + 1e-5)

        # Save cams
        np.save(os.path.join(hparams.cam_dir, os.path.splitext(os.path.basename(filename))[0] + '.npy'),
            {"keys": valid_cat,
            "cam": strided_cam,
            "high_res": highres_cam})
