#! /bin/bash
source activate ./envs
time python train_nlcd.py -bs 500 -g 0 -e 10 -p 1