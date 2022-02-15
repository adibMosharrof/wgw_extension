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

class ObjectDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cvusa_root:str,
        batch_size:int=32,
        num_workers:int=0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.cvusa_root = cvusa_root
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        img_names = list(map(
            Path,
            open(
                self.cvusa_root / "flickr_images.txt"
            ).read().strip().split("\n")[35:50],
        ))
        model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        model = model.eval()
        self.tfm = transforms.Compose([
            T.Resize([537,936])
        ])
        self.obj_dataset = ObjectDataset(img_names=img_names, cvusa_root=self.cvusa_root, model=model, tfms=self.tfm)
        dl = self.train_dataloader()
        results = []
        # for i, batch in enumerate(self.obj_dataset):
        for i, batch in enumerate(self.train_dataloader()):
            results.append(batch)
        #write results to csv file

    def train_dataloader(self):
        return DataLoader(
            self.obj_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True
        )

class ObjectDataset(Dataset):
    def __init__(self, cvusa_root:str, img_names, model=None, tfms=None) -> None:
        super().__init__()
        self.img_names = img_names
        self.cvusa_root = cvusa_root
        self.model = model
        self.tfms = tfms
        self.score_threshold = 0.8
        self.out_path = Path("out/")
    
    def __getitem__(self, index):
        # if index >= len(self):
        #     raise IndexError("index greater than data length")
        img_name = self.img_names[index]
        img = read_image(str(self.cvusa_root / img_name))
        if self.tfms:
            img = self.tfms(img)
        batch_int = torch.stack([img])
        batch = convert_image_dtype(batch_int, dtype=torch.float)
        output = self.model(batch)[0]
        mask = output['scores'] > self.score_threshold
        img_bbox = draw_bounding_boxes(batch_int[0], boxes=output['boxes'][mask], width=4)
        labels_idx = output['labels'][mask]
        img = F.to_pil_image(img_bbox)
        bbox_path = self.out_path/f'bbox_{img_name.name}'
        img.save(bbox_path)
        labels = Counter(labels_idx.tolist())
        row = {
            'labels': {**labels},
            'image_id': img_name.name,
            'bbox_path': str(bbox_path)
        }
        return row

    def __len__(self):
        return len(self.img_names)