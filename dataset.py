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

class ObjectDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cvusa_root:str,
        batch_size:int=1,
        num_workers:int=2,
        index:int=0,
        num_items:int=32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.cvusa_root = cvusa_root
        self.num_workers = num_workers
        self.index = index
        self.num_items = num_items
    
    def setup(self, stage=None):
        start = self.index*self.num_items
        img_names = list(map(
            Path,
            open(
                self.cvusa_root / "flickr_images.txt"
            ).read().strip().split("\n")[start:start+self.num_items],
        ))
        self.tfm = transforms.Compose([
            T.Resize([537,936])
        ])
        self.obj_dataset = ObjectDataset(img_names=img_names, cvusa_root=self.cvusa_root, tfms=self.tfm)


    def train_dataloader(self):
        return DataLoader(
            self.obj_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True
        )
    
    def write_csv(self, results):
        headers = ['image_id','labels']
        with open(f'out/object_detection_{self.num_items}_{self.index}.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

class ObjectDataset(Dataset):
    def __init__(self, cvusa_root:str, img_names, tfms=None) -> None:
        super().__init__()
        self.img_names = img_names
        self.cvusa_root = cvusa_root
        self.tfms = tfms
        self.score_threshold = 0.8
        self.out_path = Path("out/")
    
    def __getitem__(self, index):
        if index >= len(self) or index <0:
            raise IndexError("index greater than data length")
        img_name = self.img_names[index]
        img = read_image(str(self.cvusa_root / img_name))
        if self.tfms:
            img = self.tfms(img)
        img = convert_image_dtype(img)
        row = {
            'image_id': img_name.name,
            'image': img
        }
        return row

    def __len__(self):
        return len(self.img_names)