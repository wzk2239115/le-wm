"""
为 PushT 数据集添加语言描述

方案：
1. 使用预训练的视觉-语言模型（如 CLIP）自动生成描述
2. 手动标注关键轨迹
3. 使用模板生成描述（最简单）
"""

import h5py
import json
from pathlib import Path
from typing import List, Dict


def generate_template_descriptions(traj_data: Dict) -> str:
    """
    使用模板生成任务描述

    Args:
        traj_data: 轨迹数据，包含初始状态和目标状态

    Returns:
        任务描述
    """
    # 示例：基于目标位置生成描述
    # 实际应用中需要根据数据集的具体字段调整

    actions = traj_data["action"]  # (T, action_dim)

    # 简单启发式：分析主要运动方向
    dx = actions[:, 0].mean()
    dy = actions[:, 1].mean()

    if abs(dx) > abs(dy):
        direction = "right" if dx > 0 else "left"
        return f"Move the block to the {direction}"
    else:
        direction = "up" if dy > 0 else "down"
        return f"Move the block {direction}"


def add_language_annotations(
    hdf5_path: str,
    output_path: str,
    annotation_method: str = "template"
):
    """
    为 HDF5 数据集添加语言标注

    Args:
        hdf5_path: 原始 HDF5 文件路径
        output_path: 输出文件路径
        annotation_method: 标注方法 ("template", "clip", "manual")
    """
    with h5py.File(hdf5_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # 复制所有原始数据
        for key in f_in.keys():
            f_in.copy(key, f_out)

        # 添加语言标注
        num_trajs = len(f_in["data"])

        descriptions = []

        for i in range(num_trajs):
            traj = f_in["data"][str(i)]

            if annotation_method == "template":
                # 使用模板生成
                desc = generate_template_descriptions(traj)
            elif annotation_method == "manual":
                # 从文件加载手动标注
                with open(f"annotations/traj_{i}.txt") as f:
                    desc = f.read().strip()
            else:
                raise ValueError(f"Unknown method: {annotation_method}")

            descriptions.append(desc)

        # 保存标注
        f_out.create_dataset(
            "language",
            data=descriptions,
            dtype=h5py.string_dtype(encoding="utf-8")
        )

        # 保存元数据
        metadata = {
            "num_trajs": num_trajs,
            "annotation_method": annotation_method,
        }

        f_out.attrs["metadata"] = json.dumps(metadata)

    print(f"已创建带语言标注的数据集: {output_path}")
    print(f"轨迹数量: {num_trajs}")


def create_manual_annotations(hdf5_path: str, output_dir: str):
    """
    创建手动标注模板

    Args:
        hdf5_path: HDF5 数据集路径
        output_dir: 标注文件输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        num_trajs = len(f["data"])

    for i in range(num_trajs):
        output_path = output_dir / f"traj_{i}.txt"
        if not output_path.exists():
            output_path.write_text("# 请描述这个轨迹的任务\n")


def load_annotated_dataset(hdf5_path: str) -> List[Dict]:
    """
    加载带语言标注的数据集

    Returns:
        包含语言描述的轨迹列表
    """
    data = []

    with h5py.File(hdf5_path, "r") as f:
        num_trajs = len(f["data"])

        for i in range(num_trajs):
            traj = {
                "pixels": f["data"][str(i)]["pixels"][:],
                "action": f["data"][str(i)]["action"][:],
                "language": f["language"][i].decode("utf-8"),
            }
            data.append(traj)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入 HDF5 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 HDF5 文件")
    parser.add_argument("--method", type=str, default="template",
                       choices=["template", "manual"])
    parser.add_argument("--create-annotations", action="store_true",
                       help="创建手动标注模板")
    args = parser.parse_args()

    if args.create_annotations:
        create_manual_annotations(args.input, "annotations/")
    else:
        add_language_annotations(args.input, args.output, args.method)
