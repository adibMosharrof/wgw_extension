#! /bin/bash
source activate ./envs
time python build_object_dataset.py -si 0 -bs 8 -w 8
