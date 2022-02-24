#! /bin/bash
source activate ./envs
time python build_dataset.py -si 0 -bs 5 -w 6
