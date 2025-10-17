#!/bin/bash

# show env，查看申请的卡是否正常
nvidia-smi

# install, 安装一些依赖，docker如果存在可以不用安装
# apt-get update
# apt-get install -y libgl1-mesa-glx
# apt-get install -y libglib2.0-dev

# conda env, 激活自己的conda，注意conda安装路径
source /yinghepool/zhangshuheng/miniconda3/etc/profile.d/conda.sh
conda activate /yinghepool/zhangshuheng/miniconda3/envs/ms-swift

# run
cd /yinghepool/mabingqi/ms-swift

nproc_per_node=8
# CUDA_VISIBLE_DEVICES= \
NPROC_PER_NODE=$nproc_per_node \
VIDEO_MAX_PIXELS=114896 \
FPS_MAX_FRAMES=70 \
swift sft \
    --model /yinghepool/zhangshuheng/models/Qwen2.5-VL-3B-Instruct \
    --dataset /yinghepool/mm-data/report/tiantan/20250926-tiantan10w/AISU_level/tiantan_head_7.9w_meta_stdWindow_clean-aisu-train.jsonl \
    --val_dataset /yinghepool/mm-data/report/tiantan/20250926-tiantan10w/AISU_level/tiantan_head_7.9w_meta_stdWindow_clean-aisu-val.jsonl\
    --output_dir output/HeadReport-tiantan_Qwen2.5VL-3B_wwwl_aisu-v1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-4 \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --train_type lora \
    --use_rslora true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 81920 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --metric_for_best_model eval_token_acc \
    --greater_is_better true