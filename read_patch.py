#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script reads GeoTIFF files each of which is for one spectral 
# band of a Sentinel-2 image patch in the BigEarthNet Archive.
# 
# The script is capable of reading either  all spectral bands of one patch 
# folder (-p option) or all bands for all patches (-r option).
# 
# After reading files, Sentinel-2 image patch values can be used as numpy array 
# for further purposes.
# 
# read_patch --help can be used to learn how to use this script.
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 06 Feb 2019
# Version: 1.0.1
# Usage: read_patch.py [-h] [-p PATCH_FOLDER] [-r ROOT_FOLDER]

from __future__ import print_function
import argparse
import os

parser = argparse.ArgumentParser(
    description='This script reads the BigEarthNet image patches')
parser.add_argument('-p', '--patch_folder',
                    dest='patch_folder', help='patch folder path')
parser.add_argument('-r', '--root_folder', dest='root_folder',
                    help='root folder path contains multiple patch folders')

args = parser.parse_args()

# Checks the existence of patch folders and populate the list of patch folder paths
folder_path_list = []
if args.root_folder:
    print('INFO: -p argument will not be considered since -r argument is defined')
    if not os.path.exists(args.root_folder):
        print('ERROR: folder', args.root_folder, 'does not exist')
        exit()
    else:
        folder_path_list = [i[0] for i in os.walk(args.root_folder)][1:-1]
        if len(folder_path_list) == 0:
            print('ERROR: there is no patch directories in the root folder')
            exit()
elif not args.patch_folder:
    print('ERROR: at least one of -p and -r arguments is required')
    exit()
else:
    if not os.path.exists(args.patch_folder):
        print('ERROR: folder', args.patch_folder, 'does not exist')
        exit()
    else:
        folder_path_list = [args.patch_folder]

# Checks the existence of required python packages
gdal_existed = rasterio_existed = georasters_existed = False
try:
    import gdal
    gdal_existed = True
    print('INFO: GDAL package will be used to read GeoTIFF files')
except ImportError:
    try:
        import rasterio
        rasterio_existed = True
        print('INFO: rasterio package will be used to read GeoTIFF files')
    except ImportError:
        print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
        exit()

# Spectral band names to read related GeoTIFF files
band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# Reads spectral bands of all patches whose folder names are populated before
for patch_folder_path in folder_path_list:
    patch_folder_path = os.path.realpath(patch_folder_path)
    patch_name = os.path.basename(patch_folder_path)
    for band_name in band_names:
        # First finds related GeoTIFF path and reads values as an array
        band_path = os.path.join(
            patch_folder_path, patch_name + '_' + band_name + '.tif')
        if gdal_existed:
            band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
        elif rasterio_existed:
            band_ds = rasterio.open(band_path)
            band_data = band_ds.read(1)
        # band_data keeps the values of band band_name for the patch patch_name
        print('INFO: band', band_name, 'of patch', patch_name,
              'is ready with size', band_data.shape)
