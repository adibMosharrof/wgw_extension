
import pytorch_lightning as pl
from pathlib import Path
import csv
import torch
from argparse import ArgumentParser
from methods import WgwModel
from wgw_dataset import WgwDataModule
import tqdm.autonotebook as tqdm
from torch.distributions.poisson import Poisson
import mlflow.pytorch

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-ni", "--num_items", type=int, default=None, help="Data size"
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=2, help="Batch Size"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=2, help="Number of workers"
    )
    parser.add_argument(
        "-si", "--start_index", type=int, default=0, help="Starting index"
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="Gpu number"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Max Epochs"
    )
    parser.add_argument(
        "-ds", "--dataset", type=str, default="out/object_detection_0_100.csv", help="CSV Dataset Path"
    )
    return parser.parse_args()

def train():
    args = get_args()
    print(args)
    device_name = f'cuda:{args.gpu}'
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    wgw_dm = WgwDataModule(
        cvusa_root = Path("/u/eag-d1/data/crossview/cvusa/"),
        start_index=args.start_index,
        num_items=args.num_items,
        num_workers=args.workers,
        batch_size=args.batch_size,
        obj_dataset_csv=args.dataset
    )
    wgw_dm.setup()

    model = WgwModel() 
    # model = model.to(device)
    # model = model.eval()
    results = []

    trainer = pl.Trainer(max_epochs=args.epochs, precision=16, gpus=[args.gpu])
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        trainer.fit(model, wgw_dm)
    # trainer.fit(model, wgw_dm)

if __name__ == "__main__":
    train()