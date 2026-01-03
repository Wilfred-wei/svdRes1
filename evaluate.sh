#!/bin/bash

# Evaluation script for CnnDetTest dataset
# Modify EXP_NAME to match your trained experiment
export CUDA_VISIBLE_DEVICES=3
EXP_NAME="training_setting_1_small"

# Evaluate Stage 1 model
echo "=========================================="
echo "Evaluating Stage 1 Model"
echo "=========================================="
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

echo ""
echo "=========================================="
echo "Evaluating Stage 2 Model (with RCS Token)"
echo "=========================================="
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

echo ""
echo "=========================================="
echo "Evaluation Completed!"
echo "=========================================="
echo "Results saved in: ./check_points/${EXP_NAME}/evaluation_log.log"

