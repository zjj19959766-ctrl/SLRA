#!/bin/bash

# 确保遇到错误停止
set -e 

# 定义脚本的相对路径
SCRIPT_PATH="scripts/v1_5/train/ours-train.sh"

echo "========================================================="
echo "🚀 开始训练 [第二组]：完全体 (HOG + LoRA of LoRA)"
echo "========================================================="
export PATCH_RANK=4
export PATCH_ALPHA=1.0
export PATCH_TRAIN_EPOCHS=0.5
export HOG_LAMBDA=0.5
# 关键：从根目录调用脚本
bash $SCRIPT_PATH


echo "========================================================="
echo "🚀 开始训练 [第四组]：消融实验 (无 HOG)"
echo "========================================================="
export PATCH_RANK=4
export PATCH_ALPHA=1.0
export PATCH_TRAIN_EPOCHS=0.5
export HOG_LAMBDA=0.0
# 关键：从根目录调用脚本
bash $SCRIPT_PATH

echo "========================================================="
echo "✅ successfully finished all training runs!"
echo "========================================================="