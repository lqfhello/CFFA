#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --name CFFA \
    --batch_size 64 \
    --select_ratio 0.3 \
    --root_dir /media/ssd_2t/home/lqf/dataset \
    --output_dir run_logs \
    --margin 0.1 \
    --dataset_name CUHK-PEDES \
    --loss_names mal+mlm \
    --num_epoch 60
 