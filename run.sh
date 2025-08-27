#!/bin/sh
DEVICE_ID=1  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

apptainer exec --nv -B /sda2:/sda2 --no-home jax.sif python3 "main.py" "training.seed=$RANDOM"