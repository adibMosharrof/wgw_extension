from dataset import ObjectDataModule, ObjectDataset
import pytorch_lightning as pl
from pathlib import Path

if __name__ == "__main__":
	dm = ObjectDataModule(
		cvusa_root = Path("/u/eag-d1/data/crossview/cvusa/")

	)
	dm.setup()

	# for i, batch in enumerate(dm.train_dataloader()):
	# 	a=1