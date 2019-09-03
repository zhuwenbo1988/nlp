#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/usr/local/cuda-8.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES=""
python inference.py config/infer.config
