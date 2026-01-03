#!/bin/bash

EXP_NAME="training_setting_1_small"

# # Prepare small dataset first
# echo "Preparing small-scale balanced dataset..."
# python prepare_small_dataset.py

# train stage 1
echo "Training Stage 1..."
python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root /sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small \
    --val_data_root /sda/home/temp/weiwenfei/Datasets/progan_val_small \
    --train_classes car cat chair horse \
    --val_classes car cat chair horse \
    --training_stage 1 \
    --stage1_batch_size 16 \
    --stage1_epochs 50 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 8 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.000002 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# train stage 2
echo "Training Stage 2..."
python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root /sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small \
    --val_data_root /sda/home/temp/weiwenfei/Datasets/progan_val_small \
    --train_classes car cat chair horse \
    --val_classes car cat chair horse \
    --training_stage 2 \
    --stage1_batch_size 16 \
    --stage1_epochs 50 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 8 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.000002 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

echo "Training completed!"

# evaluate stage 1
echo "Evaluating Stage 1..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root /sda/home/temp/weiwenfei/Datasets/CnnDetTest \
    --eval_stage 1 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weights ./check_points/${EXP_NAME}/train_stage_1/model/intermediate_model_best.pth

# evaluate stage 2
echo "Evaluating Stage 2..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root /sda/home/temp/weiwenfei/Datasets/CnnDetTest \
    --eval_stage 2 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weights ./check_points/${EXP_NAME}/train_stage_2/model/model_best_val_loss.pth

echo "All tasks completed!"
