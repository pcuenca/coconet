#!/bin/bash

CHECKPOINTS_PATH="/home/pedro/code/hf/diffusers/marigold-segmentation/Marigold/output/overlapped/25_06_11-13_00_18-train_cocogold/checkpoint"
SD_PATH="/home/pedro/code/hf/diffusers/marigold-segmentation/checkpoints/stable-diffusion-2"
DATASET_PATH="/data/datasets/coco/2017"

# TODO: re-run for ensemble-runs `2 4 8` after the runs for `1` are done

python run_cocogold_eval_grid.py \
    ${DATASET_PATH} \
    ${CHECKPOINTS_PATH} \
    ${SD_PATH} \
    results.csv \
    --batch-size 64 \
    --models iter_008000 iter_018000 \
    --ensemble-runs 1 \
    --extra-trailing-model iter_018000 \
    --extra-trailing-steps 1 4 20 \
    --extra-trailing-ensemble-runs 1
