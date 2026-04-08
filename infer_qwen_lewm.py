"""
Qwen-LEWM 推理脚本
根据文字描述执行 PushT 任务
"""

import torch
import numpy as np
from qwen_lewm import QwenLEWM, create_qwen_lewm_from_checkpoint
from stable_worldmodel.data import HDF5Dataset


class QwenLEWMPolicy:
    """带语言条件的世界模型策略"""

    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def plan(self, obs, text_prompt, n_candidates=10, horizon=10):
        """
        给定观测和文本描述，规划最优动作序列

        Args:
            obs: 当前观测 (dict with pixels)
            text_prompt: 任务描述 (str)
            n_candidates: 采样候选动作数量
            horizon: 规划时域

        Returns:
            best_action: 最优动作
            all_costs: 所有候选的成本
        """
        B = 1

        # 准备初始观测
        info = {
            "pixels": obs["pixels"].unsqueeze(0).to(self.device),  # (1, H, C, H, W)
        }

        # 编码语言
        cond_emb = self.model.encode_language([text_prompt])  # (1, D)

        # 采样候选动作序列（CEM 或随机采样）
        action_dim = 2  # PushT
        frameskip = 5
        action_sequence = torch.randn(
            B, n_candidates, horizon, action_dim * frameskip,
            device=self.device
        )

        # Rollout 并评估成本
        info = self.model.rollout(info, [text_prompt], action_sequence)
        pred_emb = info["predicted_emb"]  # (1, S, T, D)

        # 计算成本（这里简化为与目标的距离）
        # 实际应用中需要定义目标状态
        goal_emb = torch.zeros(1, 1, pred_emb.size(-1), device=self.device)
        costs = torch.cdist(pred_emb[..., -1, :], goal_emb).squeeze(-1)  # (1, S)

        # 选择最优动作
        best_idx = costs.argmin(dim=-1)
        best_action = action_sequence[0, best_idx, 0]  # 第一个动作

        return best_action, costs

    def interact(self, env, text_prompt, max_steps=100):
        """
        在环境中执行任务

        Args:
            env: PushT 环境
            text_prompt: 任务描述
            max_steps: 最大交互步数

        Returns:
            trajectory: 交互轨迹
            success: 是否成功
        """
        trajectory = []
        obs = env.reset()

        for step in range(max_steps):
            # 规划最优动作
            action, costs = self.plan(obs, text_prompt)
            action_np = action.cpu().numpy()

            # 执行动作
            obs, reward, done, info = env.step(action_np)
            trajectory.append({
                "obs": obs,
                "action": action_np,
                "reward": reward,
            })

            if done:
                break

        return trajectory, info.get("success", False)


def demo_interactive():
    """交互式演示"""
    import gymnasium as gym

    # 加载模型
    policy = QwenLEWMPolicy("outputs/qwen_lewm.ckpt")

    # 创建环境（示例，需要实际的 PushT 环境）
    # env = gym.make("pusht/PushT-v0")

    # 交互循环
    while True:
        text_prompt = input("\n请输入任务描述 (q 退出): ")
        if text_prompt == "q":
            break

        print(f"执行任务: {text_prompt}")

        # obs = env.reset()
        # trajectory, success = policy.interact(env, text_prompt)

        print("任务执行完成！" if success else "任务失败")


def demo_batch_inference():
    """批量推理示例"""
    model = torch.load("outputs/qwen_lewm.ckpt", map_location="cuda")
    model.eval()

    # 批量任务描述
    text_prompts = [
        "Push the T-block to the green zone",
        "Move the block to the right",
        "Rotate the T-block 90 degrees",
    ]

    # 准备观测（示例）
    batch = {
        "pixels": torch.randn(3, 4, 3, 224, 224).cuda(),
        "action": torch.randn(3, 4, 10).cuda(),
    }

    with torch.no_grad():
        output = model(batch, text_prompts)

    print(f"预测嵌入形状: {output['pred_emb'].shape}")
    print(f"语言条件形状: {output['cond_emb'].shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "interactive"])
    parser.add_argument("--model", type=str, default="outputs/qwen_lewm.ckpt")
    args = parser.parse_args()

    if args.mode == "interactive":
        demo_interactive()
    else:
        demo_batch_inference()
