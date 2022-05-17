import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser
from methods import NlcdBaselineModel, NlcdPretrainedModel
from nlcd_dataset import NlcdDataModule
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-ni", "--num_items", type=int, default=None, help="Data size")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument(
        "-w", "--workers", type=int, default=8, help="Number of workers"
    )
    parser.add_argument(
        "-si", "--start_index", type=int, default=0, help="Starting index"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Gpu number")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Max Epochs")
    parser.add_argument("-p", "--pretrained", type=int, default=1, help="Pretrained")
    parser.add_argument("-ls", "--log_step", type=int, default=1000, help="Log Step")
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=str,
        default="lightning_logs/version_0/checkpoints/epoch=9-step=42439.ckpt",
        help="CSV Dataset Path",
    )
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        default="/localdisk1/data/cvusa_eag",
        help="CSV Dataset Path",
    )
    parser.add_argument(
        "-nlcd_r",
        "--nlcd_root",
        type=str,
        default="/localdisk1/data/nlcd_2019_land_cover",
        help="Nlcd Root Path",
    )
    parser.add_argument(
        "-or",
        "--out_root",
        type=str,
        default="out",
        help="Output Root Path",
    )
    return parser.parse_args()


def train():
    args = get_args()
    print(args)
    nlcd_dm = NlcdDataModule(
        cvusa_root=Path(args.data_root),
        nlcd_root=Path(args.nlcd_root),
        nlcd_csv_path=Path(f"out/cvusa_nlcd_{args.num_items}.csv"),
        start_index=args.start_index,
        num_items=args.num_items,
        num_workers=args.workers,
        batch_size=args.batch_size,
        out_root=Path(args.out_root),
    )
    nlcd_dm.setup()

    if args.pretrained:
        model = NlcdPretrainedModel(checkpoint_path=args.checkpoint_path)
    else:
        model = NlcdBaselineModel()

    if args.gpu == -1:
        gpu = -1
    else:
        gpu = [args.gpu]

    logger = TensorBoardLogger("lightning_logs", name="nlcd")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=16,
        gpus=gpu,
        plugins=DDPPlugin(find_unused_parameters=False),
        profiler="pytorch",
        logger=logger,
    )
    trainer.fit(model, nlcd_dm)


if __name__ == "__main__":
    train()
