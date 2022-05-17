#! /bin/bash
source activate ./envs
time python train_obj_dist.py -si 0 -bs 500 -w 8 -odsr out -e 10 -g 0 -ni 5000

