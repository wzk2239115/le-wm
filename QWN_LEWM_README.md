# Qwen-LEWM: 语言驱动的世界模型

将 **Qwen3.5B** 与 **LEWM** 世界模型融合，实现从自然语言到机器人交互的端到端控制。

## 🎯 核心思想

```
文字任务 → Qwen3.5B (冻结) → 语义嵌入 → LEWM条件输入 → PushT交互
"把 T 型积木推到绿色区域"  →  理解任务  →  条件向量  →  执行动作
```

## 📁 项目结构

```
le-wm/
├── qwen_lewm.py              # 融合模型定义
├── train_qwen_lewm.py        # 训练脚本
├── infer_qwen_lewm.py        # 推理/交互脚本
├── config/train/qwen_lewm.yaml  # 训练配置
├── scripts/
│   └── create_lang_dataset.py    # 数据集语言标注工具
└── QWEN_LEWM_README.md       # 本文档
```

## 🏗️ 架构设计

### 1. **Qwen 编码器 (冻结)**
- 提取语言语义表示
- 输出: (B, 1536) 隐藏状态
- **完全冻结，不参与训练**

### 2. **语言投影层 (可训练)**
```python
Qwen(1536) → LayerNorm → Linear(512) → GELU → Linear(192)
```
将 Qwen 嵌入投影到 LEWM 的条件空间

### 3. **条件化 LEWM**
```python
# 原始: ARPredictor(emb, action)
# 修改: ARPredictor(emb + lang_cond, action)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 Qwen
pip install transformers>=4.37.0
pip install qwen-vl-utils

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据集

**选项 A: 使用模板生成描述**
```bash
python scripts/create_lang_dataset.py \
    --input data/pusht.hdf5 \
    --output data/pusht_lang.hdf5 \
    --method template
```

**选项 B: 手动标注**
```bash
# 1. 创建标注模板
python scripts/create_lang_dataset.py \
    --input data/pusht.hdf5 \
    --create-annotations

# 2. 编辑 annotations/traj_*.txt 文件

# 3. 创建数据集
python scripts/create_lang_dataset.py \
    --input data/pusht.hdf5 \
    --output data/pusht_lang.hdf5 \
    --method manual
```

### 3. 训练模型

```bash
python train_qwen_lewm.py
```

**配置调整** (config/train/qwen_lewm.yaml):
```yaml
loader:
  batch_size: 32  # 根据 GPU 显存调整

qwen:
  model_path: "Qwen/Qwen2.5-3B-Instruct"  # 或本地路径
  freeze: true  # 冻结 Qwen
```

### 4. 推理/交互

```bash
# 批量推理
python infer_qwen_lewm.py --mode batch --model outputs/qwen_lewm.ckpt

# 交互式
python infer_qwen_lewm.py --mode interactive --model outputs/qwen_lewm.ckpt
```

## 📊 训练细节

### 显存优化
- **梯度累积**: `accumulate_grad_batches: 4`
- **Batch Size**: 32 (原 128)
- **混合精度**: `precision: bf16`

### 可训练参数
```
Qwen3.5B:  ~3B (冻结)
LEWM:      ~50M (可训练)
Projector: ~1M (可训练)
```

### 损失函数
```python
loss = pred_loss                     # 预测损失
     + λ * sigreg_loss               # 各向同性正则化
     + γ * lang_cond_loss            # 语言条件正则化
```

## 🎓 使用示例

### Python API

```python
from qwen_lewm import QwenLEWM
import torch

# 加载模型
model = QwenLEWM(qwen_model_path="Qwen/Qwen2.5-3B-Instruct")
model.load_state_dict(torch.load("outputs/qwen_lewm.ckpt"))
model.eval()

# 准备输入
batch = {
    "pixels": torch.randn(2, 4, 3, 224, 224),  # (B, T, C, H, W)
    "action": torch.randn(2, 4, 10),
}

text_prompts = [
    "Push the T-block to the green zone",
    "Move the block right"
]

# 前向传播
with torch.no_grad():
    output = model(batch, text_prompts)

print(f"预测: {output['pred_emb'].shape}")   # (2, 1, 192)
print(f"条件: {output['cond_emb'].shape}")   # (2, 192)
```

### 交互式推理

```python
from infer_qwen_lewm import QwenLEWMPolicy

# 加载策略
policy = QwenLEWMPolicy("outputs/qwen_lewm.ckpt")

# 执行任务
obs = env.reset()
action, costs = policy.plan(obs, "把积木推到目标位置")

# 执行动作
next_obs, reward, done, info = env.step(action.numpy())
```

## 🔬 实验建议

### 1. **消融实验**
- 无语言条件 (原始 LEWM)
- 可训练 Qwen (全模型微调)
- 不同投影层大小

### 2. **评估指标**
- 任务成功率
- 样本效率
- 零样本泛化能力

### 3. **数据增强**
- 使用 VLM (如 GPT-4V) 生成更丰富的描述
- 多语言支持
- 属性分解 (颜色、位置、方向)

## ⚠️ 注意事项

### 训练稳定性
1. **学习率**: Qwen 投影层使用较小学习率 (1e-5)
2. **梯度裁剪**: `gradient_clip_val: 1.0`
3. **Warmup**: 使用线性 warmup

### 推理效率
- Qwen 编码可缓存，避免重复计算
- 使用批量推理提高吞吐量

### 数据质量
- 语言描述需要清晰明确
- 避免歧义表达
- 考虑使用模板化描述提高一致性

## 📈 性能预期

基于类似工作 (RT-2, Gato)：
- **训练时间**: ~24 小时 (1x A100)
- **任务成功率**: 60-80% (PushT)
- **零样本泛化**: 可泛化到未见过的任务描述

## 🔄 迁移到其他任务

### 修改数据集
```yaml
# config/train/qwen_lewm.yaml
data:
  dataset:
    name: your_task_expert_train
    keys_to_load:
      - pixels
      - action
```

### 修改动作维度
```yaml
wm:
  action_dim: 7  # e.g., Franka Panda arm
```

## 📞 问题反馈

- 打开 Issue 报告 Bug
- Pull Request 欢迎改进
- 讨论使用 Discord

## 📄 许可证

MIT License

## 🙏 致谢

- Qwen Team (阿里巴巴)
- LeWM 原作者
- PushT 数据集
