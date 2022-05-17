from __future__ import annotations
from PIL import Image
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
import os
import pytorch_lightning as pl
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from collections import Counter
from torchvision.io import read_image
import csv
import utils
import json
from torch.utils.data import random_split
from build_nlcd_dataset import BuildNlcdDataset


class NlcdDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cvusa_root: Path = None,
        out_root: Path = None,
        nlcd_csv_path: Path = None,
        nlcd_root: Path = None,
        batch_size: int = 2,
        zoom: str = "18",
        num_workers: int = 2,
        start_index: int = 0,
        num_items: int = 1000,
        valid_pct: float = 0.05,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.cvusa_root = cvusa_root
        self.out_root = out_root
        self.nlcd_root = nlcd_root
        self.num_workers = num_workers
        self.start_index = start_index
        self.num_items = num_items
        self.nlcd_csv_path = nlcd_csv_path
        self.valid_pct = valid_pct
        self.zoom = zoom

    def setup(self, stage=None):
        if not self.nlcd_csv_path.is_file():
            bnd = BuildNlcdDataset(
                num_items=self.num_items,
                nlcd_root=self.nlcd_root,
                cvusa_root=self.cvusa_root,
                out_root=self.out_root,
                zoom=self.zoom,
            )
            bnd.build()
        data = utils.read_csv(self.nlcd_csv_path)
        self.tfm = transforms.Compose([T.Resize([256, 256])])

        if self.num_items == None:
            self.num_items = len(data)
        rows = data[self.start_index : self.start_index + self.num_items]
        valid_size = int(self.valid_pct * self.num_items)
        train_rows, valid_rows = random_split(
            rows,
            [self.num_items - valid_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_nlcd_dataset = NlcdDataset(
            data=train_rows, cvusa_root=self.cvusa_root, tfms=self.tfm
        )
        self.valid_nlcd_dataset = NlcdDataset(
            data=valid_rows, cvusa_root=self.cvusa_root, tfms=self.tfm
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_nlcd_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=utils.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_nlcd_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=utils.collate_fn,
        )


class NlcdDataset(Dataset):
    def __init__(self, cvusa_root: str, data, tfms=None) -> None:
        super().__init__()
        self.data = data
        self.cvusa_root = cvusa_root
        self.tfms = tfms

    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            raise IndexError("index greater than data length")
        row = self.data[index]
        img_name = row[0]
        lat = float(row[1])
        lon = float(row[2])
        labels = json.loads(row[3])
        label = labels[0]
        clabels = json.loads(row[4])
        if len(clabels) == 0:
            return None
        clabel = clabels[0]

        try:
            img = read_image(str(self.cvusa_root / img_name))
            if img.shape[0] != 3:
                print(f"3 dimensions not present in image with id {img_name}")
                return None

        except Exception as e:
            print(f"could not read image with id {img_name}, error msg {str(e)}")
            return None
        if self.tfms:
            img = self.tfms(img)
        img = convert_image_dtype(img)
        row = {
            # 'image_path': str(img_name),
            "aerial_img": img,
            "lat": lat,
            "lon": lon,
            "nlcd_labels": label,
            "nlcd_coarse_labels": clabel,
        }
        return row

    def __len__(self):
        return len(self.data)
