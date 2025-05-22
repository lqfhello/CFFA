#!/bin/bash
root_dir=/media/ssd_2t/home/lqf/dataset
margin=0.1
#noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=mal+mlm
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

#noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=0 \
    python train.py \
#    --noisy_rate $noisy_rate \
#    --noisy_file $noisy_file \
    --name CFFA \
    --batch_size 64 \
    --select_ratio $select_ratio \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}  \
    --num_epoch 60
 