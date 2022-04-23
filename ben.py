import matplotlib.pyplot as plt
from PIL import Image
import os
import json
root = "/u/eag-d1/data/BigEarthNet-v1.0/BigEarthNet-v1.0/"
from pyproj import CRS

dname = "S2A_MSIL2A_20170613T101031_0_45"
fname = "S2A_MSIL2A_20170613T101031_0_45_B01.tif"
path = os.path.join(root, dname, fname)
p = "/u/eag-d1/data/BigEarthNet-v1.0/BigEarthNet-v1.0/S2A_MSIL2A_20170613T101031_0_45/S2A_MSIL2A_20170613T101031_0_45_labels_metadata.json"

# with open(p) as f:
#    data = json.load(f)

with open('data.txt') as f:
   # json.dump(data, f, ensure_ascii=False, indent=4)
   data = json.load(f)

crs = CRS.from_wkt(data['projection'])
params = crs.coordinate_operation.params
lat = params[0].value
long = params[1].value
print(f'lat {lat}, long {long}')
q=1