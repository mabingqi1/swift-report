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

python -u /yinghepool/mabingqi/ms-swift/scripts/k8s_qwen2.5vl/infer.py
