# ForgeLens RCS使用指南

## 概述
本指南说明如何使用改进后的ForgeLens模型（包含RCS Token增强）进行训练和测试。

## 文件说明

### 训练脚本
- **train_setting_1.sh**: 完整的训练+评估脚本（小规模数据集）
  - 准备小规模平衡数据集
  - 训练Stage 1模型（5 epochs）
  - 训练Stage 2模型（3 epochs，含RCS Token）
  - 自动评估Stage 1和Stage 2模型

### 评估脚本
- **evaluate.sh**: 独立的评估脚本
  - 对CnnDetTest测试集进行评估
  - 分别评估Stage 1和Stage 2模型性能
  - 输出详细的准确率和AP指标

### 数据准备脚本
- **prepare_small_dataset.py**: 创建小规模平衡数据集
  - 训练集：每类100真+100假（共800张）
  - 验证集：每类50真+50假（共400张）
  - 类别：car, cat, chair, horse

## 使用方法

### 方法1: 完整训练+评估（推荐用于测试）

```bash
cd /sda/home/temp/weiwenfei/ForgeLens-res
bash train_setting_1.sh
```

这个脚本会自动：
1. 准备小规模数据集
2. 训练Stage 1模型
3. 训练Stage 2模型（含RCS Token）
4. 在CnnDetTest上评估Stage 1
5. 在CnnDetTest上评估Stage 2

### 方法2: 仅评估已训练模型

如果您已经训练好了模型，可以直接运行评估：

```bash
cd /sda/home/temp/weiwenfei/ForgeLens-res
bash evaluate.sh
```

### 方法3: 修改实验名称

如果要评估不同的实验，修改脚本中的`EXP_NAME`变量：

```bash
# 在 evaluate.sh 中修改
EXP_NAME="your_experiment_name"

# 然后运行
bash evaluate.sh
```

### 方法4: 手动运行评估

```bash
# 评估 Stage 1
python evaluate.py \
    --experiment_name training_setting_1_small \
    --eval_data_root /sda/home/temp/weiwenfei/Datasets/CnnDetTest \
    --eval_stage 1 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weights ./check_points/training_setting_1_small/train_stage_1/model/intermediate_model_best.pth

# 评估 Stage 2
python evaluate.py \
    --experiment_name training_setting_1_small \
    --eval_data_root /sda/home/temp/weiwenfei/Datasets/CnnDetTest \
    --eval_stage 2 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weights ./check_points/training_setting_1_small/train_stage_2/model/model_best_val_loss.pth
```

## 数据集路径说明

- **训练集**: `/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train`
  - 包含类别: car, cat, chair, horse
  - 每个类别有 0_real 和 1_fake 两个子目录

- **验证集**: `/sda/home/temp/weiwenfei/Datasets/progan_val`
  - 包含类别: car, cat, chair, horse
  - 每个类别有 0_real 和 1_fake 两个子目录

- **测试集**: `/sda/home/temp/weiwenfei/Datasets/CnnDetTest`
  - 包含多个生成方法生成的图像
  - 自动遍历所有子目录进行测试

## 小规模数据集路径

运行脚本后会自动创建：
- **训练集**: `/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small`
- **验证集**: `/sda/home/temp/weiwenfei/Datasets/progan_val_small`

## 输出文件

训练和评估后的文件位于 `./check_points/training_setting_1_small/`：

```
check_points/training_setting_1_small/
├── train_stage_1/
│   ├── train_stage1.log
│   └── model/
│       ├── intermediate_model_best.pth  # Stage 1最佳模型
│       └── model_epoch_*.pth            # 各epoch检查点
├── train_stage_2/
│   ├── train_stage2.log
│   ├── model/
│   │   └── model_best_val_loss.pth      # Stage 2最佳模型
│   └── evaluation_log.log               # 评估结果日志
└── evaluation_log.log                   # 整体评估日志
```

## 评估结果解读

评估脚本会测试多个生成方法，包括：
- progan, stylegan, biggan
- cyclegan, stargan, gaugan
- deepfake, seeingdark
- san, crn, imle, guided
- ldm_200, ldm_200_cfg, ldm_100
- glide_100_27, glide_50_27, glide_100_10
- dalle

每个方法会输出：
- **ACC (Accuracy)**: 分类准确率
- **AP (Average Precision)**: 平均精度

## 使用完整数据集训练

如果要使用完整数据集进行训练，修改 `train_setting_1.sh`：

```bash
# 将数据路径改为完整数据集
--train_data_root /sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train \
--val_data_root /sda/home/temp/weiwenfei/Datasets/progan_val \

# 增加 epoch 数量
--stage1_epochs 50 \
--stage2_epochs 10 \

# 注释掉数据准备步骤
# python prepare_small_dataset.py
```

## 环境要求

确保在AIDE环境中运行：

```bash
source activate AIDE
```

Python版本：3.10.16
主要依赖：PyTorch, torchvision, tensorboardX, scikit-learn

## 常见问题

### Q: 如何修改训练类别？
A: 修改 `--train_classes` 和 `--val_classes` 参数，例如：
```bash
--train_classes car cat \
--val_classes car cat \
```

### Q: 如何调整batch size？
A: 根据GPU显存修改 `--stage1_batch_size` 和 `--stage2_batch_size`

### Q: 评估时提示找不到模型怎么办？
A: 确保 `--weights` 路径正确，并检查模型文件是否存在

### Q: 如何仅运行训练不运行评估？
A: 注释掉 `train_setting_1.sh` 中的评估部分（第65-95行）

## RCS Token说明

本实现包含RCS（Residual Context Structure）Token增强：
- **Stage 1**: 使用WSGM模块训练基础检测器
- **Stage 2**: 额外使用RCS Token，利用中间层注意力特征
  - 提取5-9层的空间注意力模式
  - 聚合生成具有空间感知能力的Token
  - 与CLS Token一起输入FAFormer进行融合

这种设计在保持数据高效的同时，增强了模型对伪造痕迹的空间定位能力。
