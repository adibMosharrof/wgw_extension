import os
from torchvision.utils import make_grid, save_image, draw_bounding_boxes
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import csv
from collections import Counter

names = np.array([
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])


# cvusa_root = Path("/localdisk1/data/cvusa_eag/streetview/cutouts/24/-80/")
# cvusa_root = Path("images/")
cvusa_root = Path("/u/eag-d1/data/crossview/cvusa/flickr/39/-100")
out_path = Path('out/')
img_paths = os.listdir(cvusa_root)[:5]

# imgs = [Image.open(str(cvusa_root/i)) for i in img_paths]
imgs = [F.resize(read_image(str(cvusa_root/i)), [537,936]) for i in img_paths]

# grid = make_grid(imgs)
# imgs[0].save('test.png')

batch_int = torch.stack(imgs)
batch = convert_image_dtype(batch_int, dtype=torch.float)

model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

outputs = model(batch)

score_threshold = .8
results = []
headers = ['labels', 'image_id', 'bbox_path']
for dog_int, output, img_name in zip(batch_int, outputs, img_paths):
    mask = output['scores'] > score_threshold
    img_bbox = draw_bounding_boxes(dog_int, boxes=output['boxes'][mask], width=4)
    labels_idx = output['labels'][mask]
    img = F.to_pil_image(img_bbox)
    bbox_path = out_path/f'bbox_{img_name}'
    img.save(bbox_path)
    labels = Counter(labels_idx.tolist())
    row = {
        'labels': {**labels},
        'image_id': img_name,
        'bbox_path': str(bbox_path)
    }
    results.append(row)

with open('object_detection.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(results)