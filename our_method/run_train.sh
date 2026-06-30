#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

echo "Starting ETD Training Process..."
python train_ETD.py \
    --lr 1e-4 \
    --beta 2.0 \
    --gama 1.0 \
    --epochs 5 \
    --datanum 1000 \
    --train_dataset "./data/train_data.json" \
    --eval_dataset "./data/eval_data.json" \
    --base_model "./models/gpt" \
    --paraphraser_model "./models/t5-base" \
    --output_file "./results/output" \
    --task_name "etd_exp_v1"

echo "Training Completed."