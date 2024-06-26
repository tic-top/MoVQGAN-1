import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import os

import torch
import sys, time
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from random import randint

import movqgan.data.datasets_cdip as datasets_cdip
import movqgan.data.datasets_hwfont as datasets_hwfont
import movqgan.data.datasets_gpt_ocr as datasets_gpt_ocr
import movqgan.data.datasets_ft_ocr as datasets_ft_ocr
import movqgan.data.datasets_mix as datasets_mix

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def center_crop(image, image_size):
    width, height = image.size
    new_size = min(width, height)
    if new_size > 1.2 * image_size:
        new_size = int(1.2 * image_size)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class ImageDataset(Dataset):
    def __init__(
        self,
        mount_dir,
        image_size=512,
        infinity=False,
    ):
        self.train_dataset = datasets_cdip.CDIPDataset(mount_root=mount_dir)
        self.image_size = image_size
        self.infinity = infinity

    def __len__(self):
        return 99999999

    def __getitem__(self, item):
        image = self.train_dataset[0]['img']
        image = center_crop(image, self.image_size)
        image = image.resize(
            (self.image_size, self.image_size), resample=Image.BICUBIC, reducing_gap=1
        )
        image = np.array(image.convert("RGB"))
        image = image.astype(np.float32) / 127.5 - 1

        return np.transpose(image, [2, 0, 1])


def create_loader(batch_size, num_workers, shuffle=False, **dataset_params):
    dataset = ImageDataset(**dataset_params)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


class LightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning data class"""

    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config

    def train_dataloader(self):
        return create_loader(**self.train_config)
