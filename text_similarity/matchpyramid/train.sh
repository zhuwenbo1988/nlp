#!/bin/bash

# create log file
if [ ! -d "./log" ]; then
  mkdir ./log
fi

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/usr/local/cuda-8.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES="7"
python train.py config/train.config
