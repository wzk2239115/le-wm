# Qwen-LEWM 架构详解

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入阶段                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐              ┌──────────────┐                  │
│  │ 文字任务描述  │              │   视觉观测     │                  │
│  │ "Push T to   │              │   pixels:    │                  │
│  │  green zone" │              │  (B,T,C,H,W)  │                  │
│  └──────┬──────┘              └──────┬───────┘                  │
│         │                             │                          │
└─────────┼─────────────────────────────┼──────────────────────────┘
          │                             │
          ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         编码阶段                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────┐   ┌──────────────────────┐         │
│  │   Qwen3.5B (冻结)       │   │   ViT Encoder         │         │
│  │                         │   │   (可训练)            │         │
│  │  "Push T to green..."   │   │   pixels → emb       │         │
│  │           ↓             │   │   (B,T,768)          │         │
│  │  [CLS] token (1536)     │   │                      │         │
│  └────────────┬───────────┘   └──────────┬───────────┘         │
│               │                          │                      │
│               ▼                          ▼                      │
│  ┌────────────────────────┐   ┌──────────────────────┐         │
│  │  语言投影层 (可训练)     │   │   Projector (可训练)  │         │
│  │  1536 → 512 → 192      │   │   768 → 2048 → 192   │         │
│  └────────────┬───────────┘   └──────────┬───────────┘         │
│               │                          │                      │
│         lang_cond(B,192)            vis_emb(B,T,192)           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                     │                    │
                     └────────┬───────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         融合阶段                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────┐             │
│  │   条件拼接                                        │             │
│  │   vis_emb(B,T,192) + lang_cond(B,1,192)         │             │
│  │        ↓                                         │             │
│  │   conditioned_emb(B,T,384)  [拼接]              │             │
│  └────────────────────┬───────────────────────────┘             │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────┐             │
│  │   ARPredictor (Conditional Transformer)         │             │
│  │                                                 │             │
│  │   for each block:                               │             │
│  │     1. AdaLN-zero( conditioned_emb, lang_cond ) │             │
│  │     2. Self-Attention(因果)                     │             │
│  │     3. FFN                                      │             │
│  │                                                 │             │
│  │   输出: pred_emb(B, num_preds, 192)            │             │
│  └────────────────────┬───────────────────────────┘             │
│                       │                                          │
└───────────────────────┼──────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         输出阶段                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  pred_emb → PredictorProj → (B, num_preds, 192)                │
│                                                                   │
│  用途:                                                           │
│  1. 预测未来状态嵌入                                              │
│  2. 与目标状态计算距离                                            │
│  3. 规划最优动作序列 (CEM/MPC)                                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 模块详解

### 2.1 Qwen 编码器

```python
class QwenEncoder(nn.Module):
    def forward(self, text_prompts: List[str]) -> Tensor:
        # 1. Tokenization
        inputs = tokenizer(text_prompts, ...)

        # 2. Forward through Qwen (冻结)
        outputs = qwen.model(**inputs, output_hidden_states=True)

        # 3. 提取表示
        last_hidden = outputs.last_hidden_state  # (B, seq_len, 1536)

        # 4. 池化 (平均池化或 [CLS])
        lang_emb = pool(last_hidden, attention_mask)  # (B, 1536)

        return lang_emb
```

**关键设计**:
- 完全冻结，不参与训练
- 使用平均池化获取全局语义
- 批量推理提高效率

### 2.2 语言投影层

```python
class LangConditionProjector(nn.Module):
    def __init__(self, qwen_hidden=1536, embed_dim=192):
        self.projection = nn.Sequential(
            nn.LayerNorm(1536),       # 稳定训练
            nn.Linear(1536, 512),     # 降维
            nn.GELU(),                # 非线性
            nn.Dropout(0.1),          # 正则化
            nn.Linear(512, 192),      # 最终投影
        )

    def forward(self, lang_emb):
        return self.projection(lang_emb)  # (B, 192)
```

**关键设计**:
- 两层 MLP 逐步降维
- LayerNorm 保证训练稳定
- Dropout 防止过拟合

### 2.3 条件融合

```python
def predict(self, emb, act_emb, cond_emb):
    """
    Args:
        emb: (B, T, 192) 视觉嵌入
        act_emb: (B, T, 192) 动作嵌入
        cond_emb: (B, 192) 语言条件
    """
    B, T, D = emb.shape

    # 扩展条件到时间维度
    cond_expanded = cond_emb.unsqueeze(1).expand(B, T, D)

    # 拼接
    conditioned = torch.cat([emb, cond_expanded], dim=-1)  # (B, T, 384)

    # 预测
    pred = self.predictor(conditioned, act_emb)  # (B, num_preds, 768)

    # 投影回嵌入空间
    pred = self.predictor_proj(pred)  # (B, num_preds, 192)

    return pred
```

**关键设计**:
- 简单拼接 (vs 交叉注意力)
- 保留原始视觉信息
- 条件作为全局偏置

### 2.4 条件化 Transformer Block

```python
class ConditionalBlock(nn.Module):
    def forward(self, x, c):  # x: (B,T,D), c: (B,T,2D)
        # 1. AdaLN-zero 调制参数
        shift_msa, scale_msa, gate_msa,
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, -1)

        # 2. 条件化归一化
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        # 3. 注意力 (带门控)
        x = x + gate_msa * self.attn(x_norm)

        # 4. FFN (带门控)
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm)

        return x
```

**AdaLN-zero**:
- 自适应层归一化
- 条件决定归一化参数
- 门控控制残差连接强度

## 3. 训练流程

```python
def training_step(model, batch, text_prompts):
    # 1. 编码
    vis_emb = model.encode(batch)              # (B, T, 192)
    lang_cond = model.encode_language(text_prompts)  # (B, 192)

    # 2. 切分上下文和目标
    ctx_emb = vis_emb[:, :history_size]        # (B, 3, 192)
    tgt_emb = vis_emb[:, num_preds:]           # (B, 1, 192)

    # 3. 预测
    pred_emb = model.predict(ctx_emb, act_emb, lang_cond)  # (B, 1, 192)

    # 4. 损失
    pred_loss = F.mse_loss(pred_emb, tgt_emb)
    sigreg_loss = sigreg(vis_emb.transpose(0, 1))

    loss = pred_loss + λ * sigreg_loss

    return loss
```

## 4. 推理流程

### 4.1 批量推理

```python
# 准备输入
batch = {
    "pixels": (B, T, C, H, W),
    "action": (B, T, action_dim),
}
text_prompts = ["task 1", "task 2", ..., "task B"]

# 前向传播
output = model(batch, text_prompts)
pred_emb = output["pred_emb"]  # (B, num_preds, 192)
```

### 4.2 交互式推理

```python
# 1. 编码语言条件 (只计算一次)
lang_cond = model.encode_language([task_description])

# 2. 初始观测
obs = env.reset()
vis_emb = model.encode({"pixels": obs})

# 3. MPC/CEM 规划
for step in range(horizon):
    # 采样候选动作
    actions = sample_action_candidates(n_samples=512)

    # Rollout 评估
    costs = []
    for action_seq in actions:
        pred = model.rollout(vis_emb, lang_cond, action_seq)
        cost = compute_cost(pred, goal)
        costs.append(cost)

    # 选择最优动作
    best_action = actions[np.argmin(costs)][0]

    # 执行
    obs = env.step(best_action)
```

## 5. 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `embed_dim` | 192 | 嵌入维度 |
| `hidden_dim` | 768 | ViT-tiny 隐藏维度 |
| `qwen_hidden` | 1536 | Qwen2.5-3B 隐藏维度 |
| `history_size` | 3 | 历史窗口长度 |
| `num_preds` | 1 | 预测步数 |
| `batch_size` | 32 | 批大小 (原 128) |
| `lr` | 3e-5 | 学习率 |
| `λ_sigreg` | 0.09 | SIGReg 权重 |
| `gradient_clip` | 1.0 | 梯度裁剪 |
| `accumulate_batches` | 4 | 梯度累积 |

## 6. 内存分析

### 参数量
```
Qwen3.5B (冻结):     ~3.0B 参数
ViT-tiny:            ~8.6M 参数
ARPredictor:         ~30M 参数
Projector:           ~10M 参数
LangProjector:       ~1M 参数
─────────────────────────────
可训练:              ~50M 参数
总显存 (bf16):       ~10GB
```

### 训练显存 (A100 40GB)
```
Batch Size = 32:
  - 模型权重: ~10 GB
  - 激活值: ~8 GB
  - 梯度: ~2 GB
  - 优化器状态: ~4 GB
  ─────────────────
  总计: ~24 GB

建议: Batch Size = 32-64 (40GB GPU)
       Batch Size = 16-24 (24GB GPU)
```

## 7. 扩展性

### 7.1 支持更多模态

```python
# 添加音频条件
audio_encoder = AudioEncoder()
audio_proj = Projector(audio_dim, embed_dim)

# 多模态融合
multimodal_cond = torch.cat([
    lang_cond,      # (B, 192)
    audio_cond,     # (B, 192)
    prop_cond,      # (B, 192)  # 本体感受
], dim=-1)  # (B, 576)
```

### 7.2 支持更多任务

```python
# 修改配置文件
# config/train/qwen_lewm.yaml

data:
  dataset:
    name: franka_kitchen_train  # 新任务

wm:
  action_dim: 9  # Franka Panda 7 + gripper 2
  history_size: 5
```

### 7.3 改进架构

```python
# 使用交叉注意力替代拼接
class CrossAttentionCondition(nn.Module):
    def forward(self, vis_emb, lang_cond):
        # vis_emb: (B, T, D_vis)
        # lang_cond: (B, D_lang)

        # Lang as KV, Vis as Q
        conditioned = cross_attn(
            q=vis_emb,
            k=lang_cond.unsqueeze(1).expand(B, T, -1),
            v=lang_cond.unsqueeze(1).expand(B, T, -1),
        )
        return conditioned
```

## 8. 故障排查

### 问题 1: CUDA OOM
```yaml
# 解决方案
loader:
  batch_size: 16  # 减小
  accumulate_grad_batches: 8  # 增加

trainer:
  precision: bf16  # 使用混合精度
```

### 问题 2: 训练不稳定
```yaml
# 解决方案
optimizer:
  lr: 1e-5  # 降低学习率

loss:
  sigreg:
    weight: 0.15  # 增加正则化
```

### 问题 3: 条件不生效
```python
# 检查条件嵌入是否变化
lang_cond = model.encode_language(["task A", "task B"])
print(lang_cond.std())  # 应该 > 0.1
```

## 9. 性能优化

### 9.1 编译优化
```python
# PyTorch 2.0+
model = torch.compile(model, mode="reduce-overhead")
```

### 9.2 数据加载优化
```python
loader:
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: True
  pin_memory: True
```

### 9.3 推理优化
```python
# 缓存语言编码
@torch.no_grad()
@lru_cache(maxsize=128)
def encode_language_cached(text):
    return model.encode_language(text)
```

## 10. 参考资料

- LeWM: Learning World Models
- Qwen2.5: https://huggingface.co/Qwen
- AdaLN: https://arxiv.org/abs/2110.10845
- RT-2: https://arxiv.org/abs/2307.15818
- Gato: https://arxiv.org/abs/2205.06175
