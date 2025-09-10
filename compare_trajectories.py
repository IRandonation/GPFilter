#!/usr/bin/env python3
"""
Trajectory Data Comparison Tool (6DOF)
用于对比原始、滤波、插值轨迹数据（支持 x,y,z,roll,pitch,yaw 与速度、加速度等）
支持 3D 轨迹、姿态、速度、加速度等全方位对比
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器
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
        def load_and_pad(filepath, expected_columns=None):
            # 读取无列名的 CSV 文件
            df = pd.read_csv(filepath, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')

            if expected_columns is None:
                # 我们期望的标准列名，对应 x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw
                expected_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'vroll', 'vpitch', 'vyaw']

            num_expected = len(expected_columns)  # 12 列
            current_cols = df.shape[1]  # 当前数据有几列

            print(f"📥 加载文件: {filepath}")
            print(f"   当前数据列数: {current_cols}, 数据点数: {len(df)}")

            if current_cols < num_expected:
                # 不够的列 → 补 0 列
                num_missing = num_expected - current_cols
                print(f"   ⚠️  数据列不足，自动补 {num_missing} 个 0 列，以达到 {num_expected} 列格式")
                for i in range(num_missing):
                    df[i + current_cols] = 0.0  # 新增列，值全为 0.0
                # 截取为标准 12 列
                df = df.iloc[:, :num_expected]
            elif current_cols > num_expected:
                # 如果列太多，只取前 12 列（避免干扰，比如原始数据有额外列）
                print(f"   ⚠️  数据列过多，只取前 {num_expected} 列")
                df = df.iloc[:, :num_expected]

            # 设置标准的列名
            df.columns = expected_columns
            return df

        try:
            # === 1. 原始数据（可能只有 x,y，其他自动补0，最终12列）
            self.original_data = load_and_pad(original_file)
            print(f"✅ 原始数据加载完成，列名: {list(self.original_data.columns)}")
            print(f"   数据点数: {len(self.original_data)}")

            # === 2. 滤波数据（smoothed，同上，自动补0到12列）
            self.smoothed_data = load_and_pad(smoothed_file)
            print(f"✅ 滤波数据加载完成，列名: {list(self.smoothed_data.columns)}")
            print(f"   数据点数: {len(self.smoothed_data)}")

            # === 3. 插值数据（可能包含更多列，比如 timestamp, velocity, acceleration...）
            if interpolated_file and Path(interpolated_file).exists():
                self.interpolated_data = pd.read_csv(interpolated_file)
                self.interpolated_data = self.interpolated_data.apply(pd.to_numeric, errors='coerce')
                print(f"✅ 插值数据加载完成，列名: {list(self.interpolated_data.columns)}")
                print(f"   数据点数: {len(self.interpolated_data)}")
                print(f"   时间范围: {self.interpolated_data['timestamp'].min():.3f} ~ {self.interpolated_data['timestamp'].max():.3f} 秒")
            else:
                self.interpolated_data = None
                print("⚠️  未提供插值数据文件，或文件不存在，跳过插值数据对比")

            return True

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False

    def plot_3d_trajectories_interactive(self, save_path=None):
        """
        使用 Plotly 绘制可交互的 3D 轨迹对比图（原始 vs 滤波 vs 插值）
        并保存为 .html 文件，支持拖拽旋转、缩放
        """
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    
        fig = go.Figure()
    
        # --- 原始轨迹
        if self.original_data is not None:
            fig.add_trace(go.Scatter3d(
                x=self.original_data['x'],
                y=self.original_data['y'],
                z=self.original_data['z'],
                mode='lines+markers',
                name='Original',
                line=dict(color='blue', width=4),
                marker=dict(size=3)
            ))
    
        # --- 滤波轨迹
        if self.smoothed_data is not None:
            fig.add_trace(go.Scatter3d(
                x=self.smoothed_data['x'],
                y=self.smoothed_data['y'],
                z=self.smoothed_data['z'],
                mode='lines+markers',
                name='Smoothed',
                line=dict(color='red', width=4),
                marker=dict(size=3)
            ))
    
        # --- 插值轨迹
        if self.interpolated_data is not None:
            fig.add_trace(go.Scatter3d(
                x=self.interpolated_data['x'],
                y=self.interpolated_data['y'],
                z=self.interpolated_data['z'],
                mode='lines+markers',
                name='Interpolated',
                line=dict(color='green', width=4),
                marker=dict(size=3)
            ))
    
        # --- 图表布局
        fig.update_layout(
            title='3D Trajectory Comparison: Original vs Smoothed vs Interpolated (可拖拽旋转)',
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            width=1000,
            height=700
        )
    
        # --- 保存为交互式 HTML 文件
        if save_path is None:
            save_path = './plots/3d_trajectory_comparison_interactive.html'
    
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"📊 可交互的 3D 轨迹图已保存为 HTML 文件: {save_path}")
        print("✅ 提示：用浏览器打开此文件，即可拖拽旋转、缩放查看 3D 轨迹！")

    def plot_attitude_comparison(self, save_path=None):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Attitude Comparison: Roll, Pitch, Yaw (raw vs interpolate)', fontsize=14)

        if self.original_data is not None:
            time_orig = np.arange(len(self.original_data))
            axs[0].plot(time_orig, self.original_data['roll'], label='Original Roll', color='blue')
            axs[1].plot(time_orig, self.original_data['pitch'], label='Original Pitch', color='blue')
            axs[2].plot(time_orig, self.original_data['yaw'], label='Original Yaw', color='blue')
        if self.interpolated_data is not None:
            time_interp = self.interpolated_data['timestamp']
            axs[0].plot(time_interp, self.interpolated_data['roll'], label='Interp Roll', color='orange', linestyle='--')
            axs[1].plot(time_interp, self.interpolated_data['pitch'], label='Interp Pitch', color='orange', linestyle='--')
            axs[2].plot(time_interp, self.interpolated_data['yaw'], label='Interp Yaw', color='orange', linestyle='--')

        titles = ['Roll [rad]', 'Pitch [rad]', 'Yaw [rad]']
        for i, ax in enumerate(axs):
            ax.set_ylabel(titles[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
        axs[-1].set_xlabel('Time [s] (index or timestamp)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 姿态对比图已保存至: {save_path}")
        else:
            plt.show()

    def plot_velocity_acceleration(self, save_path=None):
        if self.interpolated_data is None:
            print("⚠️  无插值数据，跳过速度/加速度绘图")
            return

        fig, axs = plt.subplots(4, 1, figsize=(12, 16))
        fig.suptitle('Velocity, Acceleration, Angular Velocity, Angular Acceleration over Time', fontsize=14)

        time = self.interpolated_data['timestamp']

        axs[0].plot(time, self.interpolated_data['linear_velocity'], color='blue', label='Linear Velocity [m/s]')
        axs[0].set_ylabel('Vel [m/s]')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(time, self.interpolated_data['linear_acceleration'], color='red', label='Linear Accel [m/s²]')
        axs[1].set_ylabel('Accel [m/s²]')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(time, self.interpolated_data['angular_velocity'], color='green', label='Angular Vel [rad/s]')
        axs[2].set_ylabel('Ang Vel [rad/s]')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(time, self.interpolated_data['angular_acceleration'], color='purple', label='Angular Accel [rad/s²]')
        axs[3].set_ylabel('Ang Accel [rad/s²]')
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
        axs[3].set_xlabel('Time [s]')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 速度/加速度图已保存至: {save_path}")
        else:
            plt.show()

    def run_comparison(self, original_file, smoothed_file, interpolated_file=None, save_dir='./plots'):
        print("🚀 开始 6DOF 轨迹数据对比分析...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if not self.load_data(original_file, smoothed_file, interpolated_file):
            return False

        # === 1. 3D 轨迹对比图
        self.plot_3d_trajectories_interactive(save_path=f"{save_dir}/3d_trajectory_comparison.html")

        # === 2. 姿态对比图
        self.plot_attitude_comparison(save_path=f"{save_dir}/attitude_comparison.png")

        # === 3. 速度 / 加速度 / 角速度 / 角加速度
        self.plot_velocity_acceleration(save_path=f"{save_dir}/velocity_acceleration_comparison.png")

        print("✅ 对比分析完成！所有图表已保存至:", save_dir)
        return True


def main():
    parser = argparse.ArgumentParser(description='6DOF 轨迹数据对比工具（支持3D轨迹、姿态、速度、加速度）')
    parser.add_argument('--original', '-o', default='data/trajectory.csv',
                       help='原始轨迹数据文件路径（包含 x,y,z,roll,pitch,yaw,...）')
    parser.add_argument('--smoothed', '-s', default='output/smoothed_trajectory.csv',
                       help='滤波后轨迹数据文件路径')
    parser.add_argument('--interpolated', '-i', default='output/interpolated_trajectory.csv',
                       help='插值后轨迹数据文件路径')
    parser.add_argument('--save-dir', '-d', default='./plots',
                       help='图表保存目录')

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