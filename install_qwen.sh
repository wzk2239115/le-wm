#!/bin/bash
# Qwen-LEWM 环境安装脚本

set -e

echo "🚀 安装 Qwen-LEWM 依赖..."

# 1. 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 2. 安装 Transformers (支持 Qwen2.5)
echo "📦 安装 Transformers..."
pip install transformers>=4.37.0 --upgrade

# 3. 安装 Qwen VL 工具
echo "📦 安装 Qwen VL Utils..."
pip install qwen-vl-utils

# 4. 安装其他依赖
echo "📦 安装其他依赖..."
pip install einops
pip install omegaconf
pip install hydra-core
pip install lightning
pip install wandb
pip install swanlab

# 5. 检查 CUDA
echo "🔍 检查 CUDA..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA 版本: {torch.version.cuda}')"

# 6. 下载 Qwen 模型 (可选)
read -p "是否下载 Qwen2.5-3B 模型? (~6GB) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "📥 下载 Qwen2.5-3B-Instruct..."
    mkdir -p models/qwen
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen/Qwen2.5-3B-Instruct
    echo "✅ 模型已下载到 models/qwen/Qwen2.5-3B-Instruct"
fi

# 7. 测试导入
echo "🧪 测试导入..."
python << EOF
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
print("✅ Qwen 导入成功")

from qwen_lewm import QwenLEWM
print("✅ QwenLEWM 导入成功")
EOF

echo ""
echo "🎉 安装完成！"
echo ""
echo "下一步:"
echo "1. 准备数据集: python scripts/create_lang_dataset.py --help"
echo "2. 开始训练: python train_qwen_lewm.py"
echo "3. 推理测试: python infer_qwen_lewm.py --help"
