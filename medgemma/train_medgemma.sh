#!/bin/bash
# MedGemma微调训练脚本
# 使用Swift框架进行MedGemma模型微调
set -e

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/yinghepool/mabingqi/ms-swift:/yinghepool/mabingqi/ms-swift/medgemma:$PYTHONPATH"
# 导入自定义数据预处理器
python -c "import sys; sys.path.append('/yinghepool/mabingqi/ms-swift/medgemma'); import dataset"
# export ROOT_IMAGE_DIR="/yinghepool/jiawei_database/merge_mask/train_raw/train"

# 模型配置
MODEL_TYPE="medgemma_it"
MODEL_PATH="/yinghepool/mabingqi/hf_cache/models--google--medgemma-4b-it"
OUTPUT_DIR="./output/medgemma-4b-it-finetuned_slake"
# DATA_PATH="/yinghepool/mabingqi/dataset/vlm_dataset/report/cls_prompt_adapt_resolution_rg/train_medgemma-report_cls-info.json"
DATA_PATH="hf::BoKelvin/SLAKE"

# 训练配置
NUM_TRAIN_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=1e-5
GRADIENT_ACCUMULATION_STEPS=4
MAX_LENGTH=2048
WARMUP_RATIO=0.1

# Swift微调命令
PYTHONPATH=/yinghepool/mabingqi/ms-swift:/yinghepool/mabingqi/ms-swift/medgemma python -m swift.cli.sft \
    --model-type $MODEL_TYPE \
    --model $MODEL_PATH \
    --dataset $DATA_PATH \
    --custom-register-path medgemma/dataset.py \
    --num-train-epochs $NUM_TRAIN_EPOCHS \
    --per-device-train-batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --max-length $MAX_LENGTH \
    --warmup-ratio $WARMUP_RATIO \
    --output-dir $OUTPUT_DIR \
    --logging-steps 10 \
    --save-steps 500 \
    --eval-steps 500 \
    --save-only-model true \
    --dataset-num-proc 4 \

