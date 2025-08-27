#!/bin/bash
apptainer exec --nv -B /sda2:/sda2 --no-home jax.sif mlflow server --backend-store-uri sqlite:///ssl2d.db --port 5000