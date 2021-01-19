from PIL import Image
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import utils.imutils as imutils

class VOCClassification(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.data.loc[index, 'filename']).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        label = torch.FloatTensor(self.data.iloc[index, 1:].values.astype(np.int64))

        return image, label

    def __len__(self):
        return len(self.data)


class VOCClassification_KD(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        filename = self.data.loc[index, 'filename']
        image = Image.open(filename).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        label = torch.FloatTensor(self.data.iloc[index, 1:21].values.astype(np.int64))
        features = torch.FloatTensor(self.data.iloc[index, 21:].values.astype(np.int64))

        return image, label, features

    def __len__(self):
        return len(self.data)


def generate_transforms(image_size):
    train_transform = transforms.Compose([
                        imutils.RandomResizeLong(256, 512),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        imutils.normalize(),
                        imutils.RandomCrop(image_size[0]),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ])

    val_transform = transforms.Compose([
                        transforms.Resize((image_size[0], image_size[0])),
                        np.asarray,
                        imutils.normalize(),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ])

    return {"train_transforms": train_transform, "val_transforms": val_transform}


def generate_dataloaders(params, train_data, val_data, transforms, knowledge_distillation=0):
    if knowledge_distillation:
        train_dataset = VOCClassification_KD(
            data=train_data, transforms=transforms["train_transforms"]
        )
        val_dataset = VOCClassification_KD(
            data=val_data, transforms=transforms["val_transforms"]
        )
    else:
        train_dataset = VOCClassification(
            data=train_data, transforms=transforms["train_transforms"]
        )
        val_dataset = VOCClassification(
            data=val_data, transforms=transforms["val_transforms"]
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params.val_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader