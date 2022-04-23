
from pathlib import Path
import csv
from argparse import ArgumentParser
import os
import json
from re import L
from pyproj import CRS
import rasterio
import numpy as np
from pyproj import Transformer
import utils
from tqdm import tqdm


all_coarse_labels = {
    11:0,
    12:0,
    21:1,
    22:1,
    23:1,
    24:1,
    31:2,
    41:3,
    42:3,
    43:3,
    51:4,
    52:4,
    71:5,
    72:5,
    73:5,
    74:5,
    81:6,
    82:6,
    90:7,
    95:7
}

def run():
    args = get_args()
    cvusa_root = Path(args.cvusa_root)
    nlcd_root = Path(args.nlcd_root)
    
    img_path = nlcd_root / 'nlcd_2019_land_cover_l48_20210604.img'
    dataset = rasterio.open(img_path)
    img_file_paths = utils.get_img_file_names(cvusa_root)
    num_items = utils.get_num_items(args.num_items, len(img_file_paths))
    
    nlcd_csv_out_path = f'out/cvusa_nlcd_{args.num_items}.csv'
    rows = []
    for path in tqdm(img_file_paths[:num_items]):
        lat, lon = utils.get_lat_long_from_fname(path)
        aerial_path = utils.get_aerial_img_from_ground(path, args.zoom)
        labels = get_nlcd_labels(dataset, lat, lon)
        try:
            coarse_labels = get_nlcd_coarse_labels(labels)
        except KeyError:
            continue
        row = [aerial_path, lat, lon, labels, coarse_labels]
        rows.append(row)

    headers = ['aerial_path', 'latitude', 'longitude', 'nlcd_labels', 'nlcd_coarse_labels']
    utils.write_csv(headers, rows, nlcd_csv_out_path)

def get_nlcd_coarse_labels(labels):
    coarselabels = []
    for label in labels:
        cl = all_coarse_labels[label]
        if cl in coarselabels:
            continue
        coarselabels.append(cl)
    return coarselabels

def get_nlcd_labels(dataset, lat, lon):
    coords = [[lon, lat]]
    src_crs = 'EPSG:4326'
    t = Transformer.from_crs(src_crs,dataset.crs, always_xy=True)
    x,y = t.transform(lon, lat)
    values = list(rasterio.sample.sample_gen(dataset, [[x,y]]))
    return values[0]


def get_lat_long(path):
    file_exists(path, f"json file not found at location {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"json file not found at location {path}")
    with open(path) as f:
        data = json.load(f)
    crs = CRS.from_wkt(data['projection'])
    params = crs.coordinate_operation.params
    lat = params[0].value
    long = params[1].value
    return {"lat":lat, "long":long}

def get_patches_csv(path):
    patches_csv = []
    file_exists(path, f"patches csv file not found at location {path}")
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            patches_csv.append(r[0])
    return patches_csv 

def file_exists(path, msg):
    if not os.path.isfile(path):
        raise FileNotFoundError(msg)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-ni", "--num_items", type=int, default=None, help="Data size"
    )
    parser.add_argument(
        "-z", "--zoom", type=str, default="18", help="Zoom level"
    )
    parser.add_argument(
        "-nr", "--nlcd_root", type=str, default="/localdisk1/data/nlcd_2019_land_cover", help="CSV Dataset Path"
    )
    parser.add_argument(
        "-cr", "--cvusa_root", type=str, default="/localdisk1/data/cvusa_eag", help="CSV Dataset Path"
    )
    return parser.parse_args()

if __name__ == "__main__":
    run()
    
    


