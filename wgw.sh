#! /bin/bash
source activate ./envs
time python train_wgw.py -si 0 -bs 500 -w 8 -ds out/object_detection_0_None.csv -e 10 -g 0
