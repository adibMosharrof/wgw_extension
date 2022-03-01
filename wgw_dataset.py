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
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision.io import read_image
import csv

class WgwDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cvusa_root:str,
        obj_dataset_csv:str,
        batch_size:int=2,
        num_workers:int=0,
        start_index:int=0,
        num_items:int=5,
        valid_pct:float=0.05

    ):
        super().__init__()
        self.batch_size = batch_size
        self.obj_dataset_csv = obj_dataset_csv
        self.cvusa_root = cvusa_root
        self.num_workers = num_workers
        self.start_index = start_index
        self.num_items = num_items
        self.valid_pct = valid_pct
    
    def setup(self, stage=None):
        obj_ds = self._read_csv(self.obj_dataset_csv)
        
        self.tfm = transforms.Compose([
            T.ToTensor(),
            T.CenterCrop([750]),
            T.RandomCrop([512]),
            T.Resize([256, 256]),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        if self.num_items == None:
            self.num_items = len(obj_ds)
        rows = obj_ds[self.start_index:self.start_index+self.num_items]
        valid_size = int(self.valid_pct* self.num_items)
        train_rows = rows[:-valid_size]
        valid_rows = rows[-valid_size:]
        self.train_wgw_dataset = WgwDataset(data=train_rows, cvusa_root=self.cvusa_root, tfms=self.tfm)
        self.valid_wgw_dataset = WgwDataset(data=valid_rows, cvusa_root=self.cvusa_root, tfms=self.tfm)

    def train_dataloader(self):
        return DataLoader(
            self.train_wgw_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
    def val_dataloader(self):
        return DataLoader(
            self.train_wgw_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

    def _read_csv(self, path:str):
        fields = []
        rows = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            fields = next(reader)
            for r in reader:
                rows.append(r)
        return rows


class WgwDataset(Dataset):
    def __init__(self, cvusa_root:str, data=None, tfms=None) -> None:
        super().__init__()
        self.data = data
        self.cvusa_root = cvusa_root
        self.tfms = tfms
        self.zoom = '18'
    
    def _get_aerial_img(self, img_path, lat, lon):
        path_split = img_path.split('/')
        if path_split[0] == "streetview":
            root = Path(self.cvusa_root) / f'{path_split[0]}_aerial'/self.zoom/ str(int(float(lat))) / str(int(float(lon))) 
            img_name = Path(path_split[-1])
            aerial_name = "_".join(img_name.name.split('_')[:-1]) + img_name.suffix
        elif path_split[0] == "flickr":
            root = Path(self.cvusa_root) / f'{path_split[0]}_aerial'/self.zoom/ str(int(float(lat))) / str(int(float(lon))) 
            img_name = Path(path_split[-1])
            aerial_name = "_".join(img_name.name.split("_")[2:])
        aerial_path = root / aerial_name
        # return read_image(str(aerial_path))
        return np.array(Image.open(str(aerial_path)))

    def _get_labels(self, row):
        labels = [ 0 if r == '' else int(r) for r in row]
        return np.array(labels)

    def __getitem__(self, index):
        row = self.data[index]
        aerial_img= self._get_aerial_img(*row[:3])
        if self.tfms:
            aerial_img = self.tfms(aerial_img)
        labels = self._get_labels(row[3:])
        result = {
            "aerial_img": aerial_img,
            "labels_counts": labels
        }
        return result
    
    def __len__(self):
        return len(self.data)