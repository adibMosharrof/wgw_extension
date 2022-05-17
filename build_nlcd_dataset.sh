#! /bin/bash
source activate ./envs
time python build_nlcd_dataset.py -ni 5000
