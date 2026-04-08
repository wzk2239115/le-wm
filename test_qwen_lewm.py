"""
快速测试 Qwen-LEWM 模型是否正常工作
"""

import torch
import sys


def test_qwen_import():
    """测试 Qwen 导入"""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
        print("✅ Qwen2VL 导入成功")
        return True
    except Exception as e:
        print(f"❌ Qwen2VL 导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    try:
        from qwen_lewm import QwenLEWM

        # 创建小模型用于测试
        model = QwenLEWM(
            qwen_model_path="Qwen/Qwen2.5-3B-Instruct",
            embed_dim=64,  # 小维度测试
            hidden_dim=128,
            action_dim=2,
            history_size=2,
            num_preds=1,
            frameskip=5,
            depth=2,
            heads=4,
            mlp_dim=256,
        )

        print(f"✅ 模型创建成功")
        print(f"   - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    try:
        model.eval()

        # 准备测试数据
        batch = {
            "pixels": torch.randn(2, 3, 3, 224, 224),  # (B=2, T=3, C, H, W)
            "action": torch.randn(2, 3, 10),  # (B, T, action_dim * frameskip)
        }

        text_prompts = ["Test task 1", "Test task 2"]

        # 前向传播
        with torch.no_grad():
            output = model(batch, text_prompts)

        print(f"✅ 前向传播成功")
        print(f"   - pred_emb shape: {output['pred_emb'].shape}")
        print(f"   - cond_emb shape: {output['cond_emb'].shape}")

        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试显存使用"""
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，跳过显存测试")
            return

        from qwen_lewm import QwenLEWM

        print("\n📊 显存测试...")

        # 创建模型并移到 GPU
        model = QwenLEWM(
            qwen_model_path="Qwen/Qwen2.5-3B-Instruct",
            embed_dim=192,
            hidden_dim=768,
            action_dim=2,
            history_size=3,
            depth=6,
            heads=16,
            mlp_dim=2048,
        ).cuda()

        # 测量显存
        torch.cuda.reset_peak_memory_stats()

        batch = {
            "pixels": torch.randn(8, 4, 3, 224, 224).cuda(),
            "action": torch.randn(8, 4, 10).cuda(),
        }

        text_prompts = ["Test"] * 8

        with torch.no_grad():
            output = model(batch, text_prompts)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"   - 峰值显存: {peak_memory:.2f} GB")
        print(f"   - Batch size: 8")
        print(f"   - 预估最大 batch size (24GB GPU): {int(24 / peak_memory * 8)}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 显存测试失败: {e}")


def main():
    print("=" * 60)
    print("Qwen-LEWM 快速测试")
    print("=" * 60)
    print()

    # 检查环境
    print("🔍 环境检查:")
    print(f"   - PyTorch 版本: {torch.__version__}")
    print(f"   - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU 数量: {torch.cuda.device_count()}")
        print(f"   - 当前 GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 测试导入
    if not test_qwen_import():
        print("\n❌ 请先安装 Qwen:")
        print("   pip install transformers>=4.37.0")
        print("   pip install qwen-vl-utils")
        sys.exit(1)

    print()

    # 测试模型创建
    model = test_model_creation()
    if model is None:
        sys.exit(1)

    print()

    # 测试前向传播
    if not test_forward_pass(model):
        sys.exit(1)

    print()

    # 测试显存
    test_memory_usage()

    print()
    print("=" * 60)
    print("🎉 所有测试通过!")
    print("=" * 60)
    print()
    print("下一步:")
    print("1. 准备数据: python scripts/create_lang_dataset.py --help")
    print("2. 开始训练: python train_qwen_lewm.py")
    print("3. 交互测试: python infer_qwen_lewm.py --mode interactive")


if __name__ == "__main__":
    main()
