"""
Qwen3.5B + LEWM 融合模型
将语言理解与世界模型结合，实现文字描述到机器人交互的端到端控制
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from module import ARPredictor, Embedder, MLP
from jepa import JEPA


class QwenEncoder(nn.Module):
    """冻结的 Qwen3.5B 编码器，用于提取语言语义"""

    def __init__(self, model_path="Qwen/Qwen2.5-3B-Instruct", freeze=True):
        super().__init__()
        # 加载 Qwen 模型（使用 VL 版本支持多模态扩展）
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if freeze:
            # 冻结所有 Qwen 参数
            for param in self.qwen.parameters():
                param.requires_grad = False
            self.qwen.eval()

        self.hidden_size = self.qwen.config.hidden_size

    def forward(self, text prompts):
        """
        Args:
            text_prompts: List[str] 或 str，任务描述

        Returns:
            lang_emb: (B, hidden_size) 语言语义嵌入
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # Tokenize
        inputs = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.qwen.device)

        # 获取最后一层隐藏状态
        with torch.no_grad():  # 冻结推理
            outputs = self.qwen.model(**inputs, output_hidden_states=True)
            # 使用最后一个 token 的隐藏状态（类似 GPT 的方式）
            last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden_size)

            # 或者使用平均池化
            attention_mask = inputs.attention_mask.unsqueeze(-1)  # (B, seq_len, 1)
            lang_emb = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
            # lang_emb: (B, hidden_size)

        return lang_emb


class LangConditionProjector(nn.Module):
    """将 Qwen 嵌入投影到 LEWM 的条件空间"""

    def __init__(self, qwen_hidden_size, embed_dim, hidden_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(qwen_hidden_size),
            nn.Linear(qwen_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, lang_emb):
        """
        Args:
            lang_emb: (B, qwen_hidden_size)

        Returns:
            cond_emb: (B, embed_dim)
        """
        return self.projection(lang_emb)


class QwenLEWM(nn.Module):
    """Qwen + LEWM 融合模型"""

    def __init__(
        self,
        qwen_model_path="Qwen/Qwen2.5-3B-Instruct",
        embed_dim=192,
        hidden_dim=768,  # ViT-tiny 的 hidden_size
        action_dim=2,    # PushT 的动作维度
        history_size=3,
        num_preds=1,
        frameskip=5,
        **predictor_kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.history_size = history_size
        self.num_preds = num_preds

        # 1. 冻结的 Qwen 编码器
        self.qwen_encoder = QwenEncoder(qwen_model_path, freeze=True)

        # 2. 语言投影层
        self.lang_projector = LangConditionProjector(
            qwen_hidden_size=self.qwen_encoder.hidden_size,
            embed_dim=embed_dim
        )

        # 3. LEWM 组件
        from stable_pretraining import backbone

        # Vision Encoder
        self.encoder = backbone.utils.vit_hf(
            "tiny",
            patch_size=14,
            image_size=224,
            pretrained=False,
            use_mask_token=False,
        )

        # Action Encoder
        effective_act_dim = frameskip * action_dim
        self.action_encoder = Embedder(
            input_dim=effective_act_dim,
            emb_dim=embed_dim
        )

        # Projector
        self.projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

        # Predictor with language conditioning
        self.predictor = ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim + embed_dim,  # 视觉嵌入 + 语言条件
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **predictor_kwargs
        )

        self.predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    def encode(self, info):
        """编码视觉观测和动作"""
        from einops import rearrange

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # CLS token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def encode_language(self, text_prompts):
        """编码语言描述为条件向量"""
        lang_emb = self.qwen_encoder(text_prompts)  # (B, qwen_hidden)
        cond_emb = self.lang_projector(lang_emb)    # (B, embed_dim)
        return cond_emb

    def predict(self, emb, act_emb, cond_emb):
        """
        预测下一状态嵌入（带语言条件）

        Args:
            emb: (B, T, D) 视觉嵌入序列
            act_emb: (B, T, D) 动作嵌入序列
            cond_emb: (B, D) 或 (B, 1, D) 语言条件
        """
        B, T, D = emb.shape

        # 扩展语言条件到时间维度
        if cond_emb.dim() == 2:
            cond_emb = cond_emb.unsqueeze(1)  # (B, 1, D)
        cond_emb = cond_emb.expand(B, T, D)   # (B, T, D)

        # 拼接视觉嵌入和语言条件
        conditioned_emb = torch.cat([emb, cond_emb], dim=-1)  # (B, T, 2D)

        # 预测
        preds = self.predictor(conditioned_emb, act_emb)

        # 投影回嵌入空间
        from einops import rearrange
        preds = self.predictor_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=B)

        return preds

    def forward(self, info, text_prompts):
        """
        完整前向传播

        Args:
            info: dict with pixels and action
            text_prompts: List[str] 任务描述

        Returns:
            dict with embeddings and predictions
        """
        # 编码视觉和动作
        info = self.encode(info)

        # 编码语言
        cond_emb = self.encode_language(text_prompts)

        # 预测
        emb = info["emb"]
        act_emb = info["act_emb"]

        ctx_emb = emb[:, :self.history_size]
        ctx_act = act_emb[:, :self.history_size]

        pred_emb = self.predict(ctx_emb, ctx_act, cond_emb)

        info["pred_emb"] = pred_emb
        info["cond_emb"] = cond_emb

        return info

    def rollout(self, info, text_prompts, action_sequence):
        """
        带语言条件的 rollout

        Args:
            info: dict with initial pixels
            text_prompts: 任务描述
            action_sequence: (B, S, T, action_dim) 候选动作序列
        """
        from einops import rearrange

        # 编码语言条件
        cond_emb = self.encode_language(text_prompts)  # (B, D)

        # 初始状态编码
        info["action"] = action_sequence[:, :, :self.history_size]
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(-1, action_sequence.size(1), -1, -1)

        B, S = emb.shape[:2]
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()

        # 扩展条件到所有采样
        cond_emb = cond_emb.unsqueeze(1).expand(B, S, -1)  # (B, S, D)
        cond_emb = rearrange(cond_emb, "b s d -> (b s) d")

        # Rollout
        n_steps = action_sequence.size(2) - self.history_size
        act_future = rearrange(action_sequence[:, :, self.history_size:], "b s t d -> (b s) t d")

        for t in range(n_steps):
            act_emb = self.action_encoder(info["action"])
            act_emb = rearrange(act_emb, "b s ... -> (b s) ...")

            emb_trunc = emb[:, -self.history_size:]
            act_trunc = act_emb[:, -self.history_size:]

            pred_emb = self.predict(emb_trunc, act_trunc, cond_emb)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            info["action"] = torch.cat([info["action"], act_future[:, t:t+1]], dim=2)

        return info


def create_qwen_lewm_from_checkpoint(lewm_checkpoint, qwen_model_path):
    """
    从现有的 LEWM checkpoint 加载并创建融合模型

    Args:
        lewm_checkpoint: 现有的 LEWM checkpoint
        qwen_model_path: Qwen 模型路径

    Returns:
        QwenLEWM 模型（LEWM 权重从 checkpoint 加载）
    """
    model = QwenLEWM(qwen_model_path=qwen_model_path)

    # 加载 LEWM 权重
    # 注意：需要根据实际 checkpoint 格式调整
    if isinstance(lewm_checkpoint, str):
        state_dict = torch.load(lewm_checkpoint, map_location="cpu")
    else:
        state_dict = lewm_checkpoint

    # 只加载 LEWM 相关的权重（排除新添加的 Qwen 部分）
    lewm_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("qwen_"):
            lewm_state_dict[k] = v

    model.load_state_dict(lewm_state_dict, strict=False)

    return model


if __name__ == "__main__":
    # 测试代码
    model = QwenLEWM()

    # 测试前向传播
    batch = {
        "pixels": torch.randn(2, 4, 3, 224, 224),  # (B, T, C, H, W)
        "action": torch.randn(2, 4, 10),  # (B, T, action_dim * frameskip)
    }

    text_prompts = ["Push the T-block to the green zone", "Move the block to the right"]

    output = model(batch, text_prompts)
    print(f"预测嵌入形状: {output['pred_emb'].shape}")
    print(f"语言条件形状: {output['cond_emb'].shape}")
