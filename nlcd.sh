#! /bin/bash
source activate ./envs
time python train_nlcd.py -bs 500 -g 1 -e 1 -p 0