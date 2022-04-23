from pathlib import Path
import csv
import torch
import os

def get_img_file_names(cvusa_root:Path):
    cvusa_root = convert_to_path_data_type(cvusa_root)
    flickr_imgs = list(map(
        Path,
        open(
            cvusa_root / "flickr_images.txt"
        ).read().strip().split("\n"),
    ))

    streeview_imgs = list(map(
        Path,
        open(
            cvusa_root / "streetview_images.txt"
        ).read().strip().split("\n"),
    ))
    # all_img_names = [*flickr_imgs,*streeview_imgs]    
    all_img_names = [*streeview_imgs, *flickr_imgs]    
    return  all_img_names

def get_lat_long_from_fname(fpath: Path):
    if "flickr" in str(fpath):
        return map(float, fpath.stem.split("_")[2:])
    elif "streetview" in str(fpath):
        return map(float, fpath.stem.split("_")[:2]) 

def get_aerial_img_from_ground(ground_img: Path, zoom:str ="18"):
    ground_img = convert_to_path_data_type(ground_img)
    aerial_file_extension = '.jpg'
    img_type = ground_img.parts[0]
    if 'flickr' in img_type:
        name = "_".join(ground_img.stem.split('_')[-2:])
        lat = ground_img.parts[1]
        lon = ground_img.parts[2]
    elif  'streetview' in img_type:
        name = "_".join(ground_img.stem.split('_')[:-1])
        lat = ground_img.parts[2]
        lon = ground_img.parts[3]
    return os.path.join(f'{img_type}_aerial', zoom, lat, lon, name + aerial_file_extension)

def write_csv(headers, results, file_name):
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        csvwriter = csv.writer(f) 
        csvwriter.writerow(headers) 
        csvwriter.writerows(results)

def read_csv(path:str):
    fields = []
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        fields = next(reader)
        for r in reader:
            rows.append(r)
    return rows

def get_num_items(num, max_value):
    if num == None:
        return max_value
    return num

def collate_fn(batch):
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def convert_to_path_data_type(item:any):
    if not isinstance(item, Path):
        item =  Path(item)
    return item