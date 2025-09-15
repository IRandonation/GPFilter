#!/usr/bin/env python3
"""
3D Trajectory & RPY Comparison Tool
专注3D场景：
1. 3D轨迹对比（原始 vs 滤波）
2. 3D轨迹对比（原始 vs 插值）
3. RPY姿态时间序列对比（原始 vs 插值）
所有图表支持实时显示与文件保存（默认路径已配置）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('QtAgg')
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import sys

# 全局样式配置（统一图表风格）
# plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["figure.dpi"] = 120  # 默认分辨率
plt.rcParams["axes.linewidth"] = 1.0  # 坐标轴线条宽度


class Trajectory3DComparator:
    def __init__(self):
        # 存储完整数据（x,y,z,roll,pitch,yaw）- 3D轨迹需z轴，RPY需姿态角
        self.original_data = None  # 原始数据
        self.smoothed_data = None  # 滤波数据
        self.interpolated_data = None  # 插值数据
        # 时间步（用于RPY时间序列对比，匹配数据索引）
        self.original_timesteps = None
        self.interpolated_timesteps = None

    def load_data(self, original_file, smoothed_file, interpolated_file):
        """
        加载数据：要求CSV至少包含 x,y,z,roll,pitch,yaw 6列（3D轨迹需z轴，RPY需姿态角）
        无列名时默认按顺序解析，缺失列补0（但核心x/y/z不可缺失）
        """
        def load_data_core(filepath):
            # 读取CSV并处理数据类型
            df = pd.read_csv(filepath, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')  # 无效值转为NaN
            df = df.dropna(subset=[0, 1, 2])  # 必须保留x/y/z列（前3列），否则删除行

            # 检查核心列数（x/y/z为3D必需）
            if df.shape[1] < 3:
                raise ValueError(f"文件 {filepath} 列数不足！3D轨迹需至少x/y/z 3列，当前仅{df.shape[1]}列")

            # 定义标准列名（按顺序匹配：x,y,z,roll,pitch,yaw）
            std_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
            # 截取前6列（超出部分忽略），并补充缺失列（补0）
            df = df.iloc[:, :6]
            df.columns = std_columns[:df.shape[1]]
            for col in std_columns:
                if col not in df.columns:
                    df[col] = 0.0  # 缺失的姿态角补0

            # 生成时间步（按数据索引，代表帧序列）
            timesteps = np.arange(len(df))
            return df[std_columns], timesteps  # 按标准顺序返回数据

        try:
            # 1. 加载原始数据
            self.original_data, self.original_timesteps = load_data_core(original_file)
            print(f"✅ 原始数据加载完成 | 数据点数: {len(self.original_data)} | 包含列: x,y,z,roll,pitch,yaw")

            # 2. 加载滤波数据
            self.smoothed_data, self.smoothed_timesteps = load_data_core(smoothed_file)
            print(f"✅ 滤波数据加载完成 | 数据点数: {len(self.smoothed_data)} | 包含列: x,y,z,roll,pitch,yaw")

            # 3. 加载插值数据
            self.interpolated_data, self.interpolated_timesteps = load_data_core(interpolated_file)
            print(f"✅ 插值数据加载完成 | 数据点数: {len(self.interpolated_data)} | 包含列: x,y,z,roll,pitch,yaw")

            return True

        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False

    def plot_3d_trajectory_pair(self, data1, label1, data2, label2, title, save_path=None):
        """
        通用3D轨迹对比绘图函数（支持任意两组数据对比）
        :param data1: 第一组数据（如原始数据）
        :param label1: 第一组数据标签
        :param data2: 第二组数据（如滤波/插值数据）
        :param label2: 第二组数据标签
        :param title: 图表标题
        :param save_path: 保存路径（None则实时显示）
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 样式配置（区分两组数据）
        style1 = {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.5, "alpha": 0.9}
        style2 = {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.5, "alpha": 0.9}
        marker_style = {"s": 80, "edgecolor": "white", "linewidth": 1.0}  # 起点/终点标记样式

        # 绘制两组3D轨迹
        ax.plot(data1['x'], data1['y'], data1['z'], label=label1, **style1)
        ax.plot(data2['x'], data2['y'], data2['z'], label=label2, **style2)

        # 标记轨迹起点（圆形）和终点（星形）- 仅标记第一组数据（避免混乱）
        # 起点
        ax.scatter(data1['x'].iloc[0], data1['y'].iloc[0], data1['z'].iloc[0],
                   c=style1["color"], marker="o", label=f"{label1} begin", **marker_style)
        # 终点
        ax.scatter(data1['x'].iloc[-1], data1['y'].iloc[-1], data1['z'].iloc[-1],
                   c=style1["color"], marker="*", label=f"{label1} end", **marker_style)

        # 3D图表关键配置
        ax.set_xlabel("X Axis [m]", fontsize=11, labelpad=10)
        ax.set_ylabel("Y Axis [m]", fontsize=11, labelpad=10)
        ax.set_zlabel("Z Axis [m]", fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, linestyle="--", alpha=0.3)

        # 调整视角（默认45°仰角，便于观察3D轨迹）
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 3D轨迹图已保存: {save_path}")
        else:
            plt.show()

    def plot_rpy_comparison(self, save_path=None):
        """
        RPY姿态对比（处理插值数据10倍点数的情况）
        使用归一化时间轴（0到1.0），使原始和插值数据在相同时间范围内显示
        """
        # 创建3行1列的子图
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1, 1, 1.2])
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
        
        # 总标题
        fig.suptitle("RPY Attitude Comparison (Original vs Interpolated)", 
                     fontsize=14, fontweight="bold", y=0.95)

        # 姿态角配置
        rpy_config = [
            ("roll", "Roll [rad]", "#2ca02c"),
            ("pitch", "Pitch [rad]", "#d62728"),
            ("yaw", "Yaw [rad]", "#9467bd")
        ]

        # 绘制每个姿态角的对比曲线
        for i, (col, label, color) in enumerate(rpy_config):
            ax = axes[i]
            
            # 原始数据（点数少，实线）
            ax.plot(self.original_timesteps, self.original_data[col],
                    color=color, linestyle="-", linewidth=1.5, alpha=0.9, label="Original")
            
            # 插值数据（点数多10倍，虚线）
            ax.plot(self.smoothed_timesteps, self.smoothed_data[col],
                    color=color, linestyle="-.", linewidth=2.5, alpha=0.8, label="smoothed")

            # 子图配置
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=9, loc="upper right")
            ax.spines["top"].set_visible(False)  # 移除顶部边框
            
            # 设置x轴范围（0到1.0，归一化时间）
            # ax.set_xlim(0, 1.0)

        # 最后一个子图添加x轴标签（归一化时间）
        axes[-1].set_xlabel("Normalized Time (0 to 1.0)", fontsize=11)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.08)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 RPY对比图已保存: {save_path}")
        else:
            plt.show()

    def run_full_comparison(self, original_file, smoothed_file, interpolated_file, save_dir='./plots'):
        """
        执行完整对比流程：加载数据 → 绘制3D对比图 → 绘制RPY对比图
        :param save_dir: 图表保存目录（自动创建）
        """
        # 创建保存目录（不存在则创建）
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 图表保存目录: {save_dir.absolute()}")

        # 加载数据（失败则退出）
        if not self.load_data(original_file, smoothed_file, interpolated_file):
            print("❌ 数据加载失败，终止对比流程")
            return

        # 1. 绘制3D原始 vs 滤波轨迹
        self.plot_3d_trajectory_pair(
            data1=self.original_data,
            label1="Original",
            data2=self.smoothed_data,
            label2="Smoothed",
            title="3D Trajectory: Original vs Smoothed",
            save_path=save_dir / "3d_original_vs_smoothed.png"
        )

        # 2. 绘制3D原始 vs 插值轨迹
        self.plot_3d_trajectory_pair(
            data1=self.original_data,
            label1="Original",
            data2=self.interpolated_data,
            label2="Interpolated",
            title="3D Trajectory: Original vs Interpolated",
            save_path=save_dir / "3d_original_vs_interpolated.png"
        )

        # 3. 绘制RPY原始 vs 插值对比
        self.plot_rpy_comparison(
            save_path=save_dir / "rpy_original_vs_interpolated.png"
        )

        print("\n🎉 所有3D轨迹与RPY对比图表生成完成！")


def main():
    # 命令行参数解析（关键修改：给3个核心参数设置默认路径）
    parser = argparse.ArgumentParser(description="3D Trajectory & RPY Comparison Tool（默认路径已配置）")
    parser.add_argument('--original', '-o', 
                        default='/home/chen/Documents/GPMPFilter/data/trajectory.csv',  # 默认原始数据路径
                        help='原始3D轨迹数据文件路径（默认：/home/chen/Documents/GPMPFilter/data/trajectory.csv）')
    parser.add_argument('--smoothed', '-s', 
                        default='/home/chen/Documents/GPMPFilter/output/smoothed_trajectory.csv',  # 默认滤波数据路径
                        help='滤波后3D轨迹数据文件路径（默认：/home/chen/Documents/GPMPFilter/output/smoothed_trajectory.csv）')
    parser.add_argument('--interpolated', '-i', 
                        default='/home/chen/Documents/GPMPFilter/output/interpolated_trajectory.csv',  # 默认插值数据路径
                        help='插值后3D轨迹数据文件路径（默认：/home/chen/Documents/GPMPFilter/output/interpolated_trajectory.csv）')
    parser.add_argument('--save-dir', '-d', default='/home/chen/Documents/GPMPFilter/plots', 
                        help='图表保存目录（默认：/home/chen/Documents/GPMPFilter/plots）')

    args = parser.parse_args()

    # 初始化对比器并执行
    comparator = Trajectory3DComparator()
    comparator.run_full_comparison(
        original_file=args.original,
        smoothed_file=args.smoothed,
        interpolated_file=args.interpolated,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()