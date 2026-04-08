"""
Qwen-LEWM 训练脚本
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from qwen_lewm import QwenLEWM, create_qwen_lewm_from_checkpoint
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def qwen_lewm_forward(self, batch, stage, cfg):
    """带语言条件的训练前向传播"""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # 生成语言提示（可以从数据集中加载，这里使用简单的占位符）
    # TODO: 从数据集中加载真实的语言描述
    batch_size = batch["pixels"].size(0)
    text_prompts = ["Push the T-block to the target"] * batch_size

    # Replace NaN values
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # 编码视觉和语言
    output = self.model.encode(batch)
    cond_emb = self.model.encode_language(text_prompts)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    # 使用语言条件进行预测
    pred_emb = self.model.predict(ctx_emb, ctx_act, cond_emb)

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # 语言条件正则化（鼓励条件嵌入与任务相关）
    # 这里可以添加对比学习或其他语言条件损失
    lang_cond_loss = 0.0
    if cfg.loss.lang_cond.weight > 0:
        # 示例：条件嵌入的多样性损失
        lang_cond_loss = -cond_emb.std(dim=0).mean()  # 最大化方差

    output["lang_cond_loss"] = lang_cond_loss
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + cfg.loss.lang_cond.weight * lang_cond_loss
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="qwen_lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    # 创建 Qwen-LEWM 模型
    model = QwenLEWM(
        qwen_model_path=cfg.qwen.model_path,
        embed_dim=cfg.wm.embed_dim,
        hidden_dim=768,  # ViT-tiny
        action_dim=cfg.wm.action_dim,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        frameskip=cfg.wm.frameskip,
        **cfg.predictor
    )

    # 如果有预训练的 LEWM checkpoint，加载权重
    if cfg.get("lewm_checkpoint", None):
        print(f"Loading LEWM weights from {cfg.lewm_checkpoint}")
        model = create_qwen_lewm_from_checkpoint(cfg.lewm_checkpoint, cfg.qwen.model_path)

    # 优化器（只优化非 Qwen 部分）
    qwen_params = []
    lewm_params = []

    for name, param in model.named_parameters():
        if "qwen_encoder" in name:
            qwen_params.append(param)
        else:
            lewm_params.append(param)

    print(f"Qwen 参数: {sum(p.numel() for p in qwen_params):,}")
    print(f"LEWM 参数: {sum(p.numel() for p in lewm_params):,}")

    optimizers = {
        'model_opt': {
            "modules": list(model.parameters()),  # 所有参数都会被优化，但 Qwen 的 requires_grad=False
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(qwen_lewm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif OmegaConf.select(cfg, "swanlab.enabled"):
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        logger = SwanLabLogger(**OmegaConf.to_container(cfg.swanlab.config, resolve=True))
        try:
            logger.log_hyperparams(OmegaConf.to_container(cfg))
        except Exception:
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
