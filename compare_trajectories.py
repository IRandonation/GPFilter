#!/usr/bin/env python3
"""
2D Trajectory Comparison Tool (仅 x, y)
支持对比 原始 / 滤波 / 插值 三类轨迹数据（2D：x, y）
并计算 插值数据中的 线速度、角速度、线加速度、角加速度（基于时间导数）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


class TrajectoryComparator:
    def __init__(self):
        self.original_data = None
        self.smoothed_data = None
        self.interpolated_data = None

    def load_data(self, original_file, smoothed_file, interpolated_file):
        def load_and_pad(filepath, expected_columns=6):  # 只需要 x, y, roll, pitch, yaw, 但为了兼容，我们最小化
            df = pd.read_csv(filepath, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            current_cols = df.shape[1]

            # 我们假设你的实际数据至少有 x, y（2D），可能还有 roll, pitch, yaw（但可以没有）
            # 我们最小化兼容：只加载 x, y，其他补0
            expected_simple = ['x', 'y']  # 我们只关心 x, y
            if current_cols < 2:
                raise ValueError(f"数据至少需要包含 x, y 两列，但当前只有 {current_cols} 列")

            # 只取前两列（x, y），其他补0以兼容后续处理
            df = df.iloc[:, :2]  # 只取 x, y
            df.columns = ['x', 'y']

            # 为了兼容后面的统一列名操作，我们补充其他列为 0
            for col in ['roll', 'pitch', 'yaw']:
                df[col] = 0.0

            # 重新按完整 6 列顺序命名（但只用 x, y）
            df = df[['x', 'y', 'roll', 'pitch', 'yaw']]
            return df

        try:
            # === 1. 原始数据（只取 x, y，补其他为0，兼容12列）
            self.original_data = load_and_pad(original_file)
            print(f"✅ 原始数据加载完成（2D: x, y），数据点数: {len(self.original_data)}")

            # === 2. 滤波数据（smoothed，只取 x, y，补其他为0）
            self.smoothed_data = load_and_pad(smoothed_file)
            print(f"✅ 滤波数据加载完成（2D: x, y），数据点数: {len(self.smoothed_data)}")

            # === 3. 插值数据（只取 x, y，其他补0）
            self.interpolated_data = load_and_pad(interpolated_file)
            print(f"✅ 插值数据加载完成（2D），数据点数: {len(self.interpolated_data)}")

            return True

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False


    def plot_2d_trajectories(self, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

        # 颜色和线型配置
        style_map = {
            "Original": {"color": "tab:blue", "linestyle": "-", "alpha": 0.9, "linewidth": 2},
            "Smoothed": {"color": "tab:green", "linestyle": "--", "alpha": 0.9, "linewidth": 2},
            "Interpolated": {"color": "tab:red", "linestyle": "-.", "alpha": 0.9, "linewidth": 2},
        }

        # 绘制轨迹
        if self.original_data is not None:
            ax.plot(self.original_data['x'], self.original_data['y'], label="Original", **style_map["Original"])
            # 标记起点终点
            ax.scatter(self.original_data['x'].iloc[0], self.original_data['y'].iloc[0],
                       c=style_map["Original"]["color"], marker="o", s=60, label="Start (Original)")
            ax.scatter(self.original_data['x'].iloc[-1], self.original_data['y'].iloc[-1],
                       c=style_map["Original"]["color"], marker="X", s=70, label="End (Original)")

        if self.smoothed_data is not None:
            ax.plot(self.smoothed_data['x'], self.smoothed_data['y'], label="Smoothed", **style_map["Smoothed"])

        if self.interpolated_data is not None:
            ax.plot(self.interpolated_data['x'], self.interpolated_data['y'], label="Interpolated", **style_map["Interpolated"])

        # 坐标轴 & 样式
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_title("2D Trajectory Comparison", fontsize=14, weight="bold")
        ax.axis("equal")  # 保持比例一致
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10, loc="best", frameon=True)

        # 优化边距
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 2D轨迹对比图已保存: {save_path}")
        else:
            plt.show()


    def run_comparison(self, original_file, smoothed_file, interpolated_file=None, save_dir='./plots'):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if not self.load_data(original_file, smoothed_file, interpolated_file):
            return

        self.plot_2d_trajectories(save_path=f"{save_dir}/2d_trajectory_comparison.png")  # 修改文件名

def main():
    parser = argparse.ArgumentParser(description='2D 轨迹对比工具：支持原始/滤波/插值对比 + 速度加速度计算（仅 x, y）')
    parser.add_argument('--original', '-o', default='data/trajectory.csv', help='原始轨迹数据（x, y）')
    parser.add_argument('--smoothed', '-s', default='output/smoothed_trajectory.csv', help='滤波后轨迹数据（x, y）')
    parser.add_argument('--interpolated', '-i', default='output/interpolated_trajectory.csv', help='插值轨迹数据（带timestamp, x, y）')
    parser.add_argument('--save-dir', '-d', default='./plots', help='保存图像的目录')

    args = parser.parse_args()

    comparator = TrajectoryComparator()
    comparator.run_comparison(
        original_file=args.original,
        smoothed_file=args.smoothed,
        interpolated_file=args.interpolated,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()