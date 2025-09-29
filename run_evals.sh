#!/bin/bash

checkpoints_path="/home/pedro/code/hf/diffusers/marigold-segmentation/Marigold/output/overlapped/25_06_11-13_00_18-train_cocogold/checkpoint"
sd_path="/home/pedro/code/hf/diffusers/marigold-segmentation/checkpoints/stable-diffusion-2"
dataset_path="/data/datasets/coco/2017"

python run_cocogold_eval_grid.py \
    dataset_path \
    checkpoints_path \
    sd_path \
    results.csv \
    --batch-size 32 \
    --models iter_008000 iter_018000 \
    --ensemble-runs 1 2 4 8 \
    --extra-trailing-model iter_018000 \
    --extra-trailing-steps 1 4 20 \
    --extra-trailing-ensemble-runs 1
