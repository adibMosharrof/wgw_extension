from dataset import ObjectDataModule, ObjectDataset
import pytorch_lightning as pl
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
import csv
from collections import Counter
import torch

def write_csv(headers, results):
		with open(f'out/object_detection.csv', 'w', encoding='UTF8', newline='') as f:
			csvwriter = csv.writer(f) 
			csvwriter.writerow(headers) 
			csvwriter.writerows(results)
		

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dm = ObjectDataModule(
		cvusa_root = Path("/u/eag-d1/data/crossview/cvusa/"),
		index=0
	)
	dm.setup()
	score_threshold = 0.8
	num_labels = 91
	model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
	model = model.to(device)
	model = model.eval()
	results = []
	for batch in dm.train_dataloader():
		
		# output = model(batch['image'])
		d = batch['image'].to(device)
		output = model(d)
		for out, img_id in zip(output, batch['image_id']):
			all_labels = [""]*num_labels
			mask = out['scores'] > score_threshold
			labels_idx = out['labels'][mask]
			labels = Counter(labels_idx.tolist())
			for index, count in labels.items():
				all_labels[index] = count	
			all_labels.insert(0,img_id)
			results.append(all_labels)
	headers = ['image_id']
	headers.extend(list(range(num_labels)))
	write_csv(headers, results)