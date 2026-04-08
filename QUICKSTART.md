# Qwen-LEWM 快速参考

## 📋 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `qwen_lewm.py` | 338 | 核心模型 |
| `train_qwen_lewm.py` | 207 | 训练脚本 |
| `infer_qwen_lewm.py` | 164 | 推理脚本 |
| `test_qwen_lewm.py` | 180 | 测试脚本 |
| `scripts/create_lang_dataset.py` | 158 | 数据处理 |
| **总计** | **1047** | **完整实现** |

## 🚀 三步开始

### 1️⃣ 安装 (5分钟)
```bash
bash install_qwen.sh
```

### 2️⃣ 测试 (2分钟)
```bash
python test_qwen_lewm.py
```

### 3️⃣ 训练 (24小时)
```bash
python train_qwen_lewm.py
```

## 🎯 核心架构

```python
# 文本 → Qwen (冻结) → 投影 → LEWM (可训练) → 动作
text → QwenEncoder(3B, frozen) → Projector(1M)
     → lang_cond(192) → ARPredictor(50M) → action
```

## 📊 关键数字

| 指标 | 值 |
|------|-----|
| 总参数 | ~3.05B |
| 可训练 | ~51M |
| 显存 (训练) | ~24GB |
| 显存 (推理) | ~10GB |
| Batch Size | 32 |
| 训练时间 | ~24h (A100) |

## 🔧 常用命令

### 训练
```bash
# 默认配置
python train_qwen_lewm.py

# 自定义配置
python train_qwen_lewm.py loader.batch_size=16 optimizer.lr=1e-5

# 从 checkpoint 继续
python train_qwen_lewm.py +ckpt=outputs/qwen_lewm.ckpt
```

### 推理
```bash
# 批量
python infer_qwen_lewm.py --mode batch --model outputs/qwen_lewm.ckpt

# 交互
python infer_qwen_lewm.py --mode interactive
```

### 数据处理
```bash
# 模板生成
python scripts/create_lang_dataset.py \
    --input data/pusht.hdf5 \
    --output data/pusht_lang.hdf5 \
    --method template

# 手动标注
python scripts/create_lang_dataset.py --input data/pusht.hdf5 --create-annotations
# 编辑 annotations/*.txt
python scripts/create_lang_dataset.py --input data/pusht.hdf5 --output data/pusht_lang.hdf5 --method manual
```

## 💻 代码示例

### 快速测试
```python
from qwen_lewm import QwenLEWM
import torch

# 创建模型
model = QwenLEWM(qwen_model_path="Qwen/Qwen2.5-3B-Instruct")

# 测试
batch = {"pixels": torch.randn(2, 4, 3, 224, 224),
         "action": torch.randn(2, 4, 10)}
output = model(batch, ["task 1", "task 2"])
print(output["pred_emb"].shape)  # (2, 1, 192)
```

### 推理
```python
from infer_qwen_lewm import QwenLEWMPolicy

policy = QwenLEWMPolicy("outputs/qwen_lewm.ckpt")
action, costs = policy.plan(obs, "把积木推到目标")
```

## 🐛 故障排除

| 问题 | 解决方案 |
|------|----------|
| CUDA OOM | `batch_size=16`, `accumulate_grad_batches=8` |
| 训练不稳定 | `lr=1e-5`, `sigreg.weight=0.15` |
| Qwen 下载失败 | 使用本地路径或 modelscope |
| 条件不生效 | 检查 `lang_cond.std() > 0.1` |

## 📖 文档导航

- **新手**: 先看 `QWEN_LEWM_README.md`
- **深入**: 再看 `ARCHITECTURE.md`
- **总结**: 最后看 `IMPLEMENTATION_SUMMARY.md`

## 🎓 学习路径

```
第1天: 安装 + 测试 (理解流程)
第2天: 准备数据 + 训练 (开始训练)
第3-7天: 监控训练 (调整超参数)
第8天: 推理测试 (验证效果)
第9-10天: 改进优化 (尝试新想法)
```

## 🌟 高级技巧

### 1. 使用本地 Qwen
```python
model = QwenLEWM(qwen_model_path="/path/to/qwen")
```

### 2. 冻结更多层
```python
for name, param in model.named_parameters():
    if "lang_projector" not in name:  # 只训练投影层
        param.requires_grad = False
```

### 3. 自定义损失
```python
loss = pred_loss + sigreg_loss + contrastive_loss
```

### 4. 多 GPU
```bash
python train_qwen_lewm.py trainer.devices=4
```

## 🔗 相关资源

- [Qwen2.5](https://huggingface.co/Qwen)
- [LeWM 论文](https://arxiv.org/...)
- [PushT 数据集](https://github.com/...)

## 💬 获取帮助

- GitHub Issues
- Discussion Forum
- Email Support

---

**开始你的语言驱动机器人之旅！🚀**
