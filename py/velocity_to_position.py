#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从滤波后的速度数据反推位置，并与实际位置进行对比分析

功能：
1. 从滤波后的速度数据（60Hz）积分计算位置
2. 与实际滤波位置进行对比
3. 分别对xyz位置和rpy姿态进行对比绘图
4. 计算积分误差和统计分析

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import integrate

class VelocityToPositionAnalyzer:
    """从速度反推位置并进行对比分析的类"""
    
    def __init__(self):
        self.smoothed_data = None
        self.integrated_data = None
        self.sampling_rate = 60.0  # 60Hz
        self.dt = 1.0 / self.sampling_rate
        
    def load_smoothed_data(self, filepath):
        """
        加载滤波后的数据（18列格式）
        包含位置、速度、加速度和姿态信息
        """
        try:
            # 尝试读取CSV，可能有header
            try:
                df = pd.read_csv(filepath)
                if df.shape[1] >= 18:
                    # 有header的情况，重命名列
                    df = df.iloc[:, :18]
                    df.columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
                                 'rx', 'ry', 'rz', 'vrx', 'vry', 'vrz', 'arx', 'ary', 'arz']
                else:
                    raise ValueError("Column count insufficient")
            except:
                # 无header的情况
                df = pd.read_csv(filepath, header=None)
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(subset=[0, 1, 2])
                
                if df.shape[1] < 18:
                    raise ValueError(f"Smoothed data file {filepath} needs 18 columns, got {df.shape[1]}")
                
                df = df.iloc[:, :18]
                df.columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
                             'rx', 'ry', 'rz', 'vrx', 'vry', 'vrz', 'arx', 'ary', 'arz']
            
            # 添加时间戳
            df['timestamp'] = np.arange(len(df)) * self.dt
            
            self.smoothed_data = df
            print(f"✅ 滤波数据加载成功 | 数据点: {len(df)} | 采样率: {self.sampling_rate}Hz")
            return True
            
        except FileNotFoundError:
            print(f"❌ 错误：找不到文件 {filepath}")
            return False
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def integrate_velocity_to_position(self, initial_position=None):
        """
        从速度积分计算位置
        
        参数:
        - initial_position: 初始位置 [x0, y0, z0, rx0, ry0, rz0]，如果为None则使用滤波数据的第一个点
        """
        if self.smoothed_data is None:
            print("❌ 错误：请先加载滤波数据")
            return False
        
        # 获取速度数据
        vx = self.smoothed_data['vx'].values
        vy = self.smoothed_data['vy'].values
        vz = self.smoothed_data['vz'].values
        vrx = self.smoothed_data['vrx'].values
        vry = self.smoothed_data['vry'].values
        vrz = self.smoothed_data['vrz'].values
        
        # 设置初始位置
        if initial_position is None:
            # 使用滤波数据的第一个点作为初始位置
            x0 = self.smoothed_data['x'].iloc[0]
            y0 = self.smoothed_data['y'].iloc[0]
            z0 = self.smoothed_data['z'].iloc[0]
            rx0 = self.smoothed_data['rx'].iloc[0]
            ry0 = self.smoothed_data['ry'].iloc[0]
            rz0 = self.smoothed_data['rz'].iloc[0]
        else:
            x0, y0, z0, rx0, ry0, rz0 = initial_position
        
        # 使用累积梯形积分计算位置
        x_integrated = np.zeros(len(vx))
        y_integrated = np.zeros(len(vy))
        z_integrated = np.zeros(len(vz))
        rx_integrated = np.zeros(len(vrx))
        ry_integrated = np.zeros(len(vry))
        rz_integrated = np.zeros(len(vrz))
        
        # 设置初始值
        x_integrated[0] = x0
        y_integrated[0] = y0
        z_integrated[0] = z0
        rx_integrated[0] = rx0
        ry_integrated[0] = ry0
        rz_integrated[0] = rz0
        
        # 累积积分
        for i in range(1, len(vx)):
            x_integrated[i] = x_integrated[i-1] + vx[i] * self.dt
            y_integrated[i] = y_integrated[i-1] + vy[i] * self.dt
            z_integrated[i] = z_integrated[i-1] + vz[i] * self.dt
            rx_integrated[i] = rx_integrated[i-1] + vrx[i] * self.dt
            ry_integrated[i] = ry_integrated[i-1] + vry[i] * self.dt
            rz_integrated[i] = rz_integrated[i-1] + vrz[i] * self.dt
        
        # 创建积分结果数据框
        self.integrated_data = pd.DataFrame({
            'timestamp': self.smoothed_data['timestamp'],
            'x_integrated': x_integrated,
            'y_integrated': y_integrated,
            'z_integrated': z_integrated,
            'rx_integrated': rx_integrated,
            'ry_integrated': ry_integrated,
            'rz_integrated': rz_integrated,
            'x_actual': self.smoothed_data['x'],
            'y_actual': self.smoothed_data['y'],
            'z_actual': self.smoothed_data['z'],
            'rx_actual': self.smoothed_data['rx'],
            'ry_actual': self.smoothed_data['ry'],
            'rz_actual': self.smoothed_data['rz']
        })
        
        # 计算误差
        self.integrated_data['x_error'] = self.integrated_data['x_integrated'] - self.integrated_data['x_actual']
        self.integrated_data['y_error'] = self.integrated_data['y_integrated'] - self.integrated_data['y_actual']
        self.integrated_data['z_error'] = self.integrated_data['z_integrated'] - self.integrated_data['z_actual']
        self.integrated_data['rx_error'] = self.integrated_data['rx_integrated'] - self.integrated_data['rx_actual']
        self.integrated_data['ry_error'] = self.integrated_data['ry_integrated'] - self.integrated_data['ry_actual']
        self.integrated_data['rz_error'] = self.integrated_data['rz_integrated'] - self.integrated_data['rz_actual']
        
        print(f"✅ 速度积分完成 | 数据点: {len(self.integrated_data)}")
        return True
    
    def calculate_statistics(self):
        """计算积分误差的统计信息"""
        if self.integrated_data is None:
            print("❌ 错误：请先进行速度积分")
            return None
        
        stats = {}
        error_columns = ['x_error', 'y_error', 'z_error', 'rx_error', 'ry_error', 'rz_error']
        
        for col in error_columns:
            error_data = self.integrated_data[col]
            stats[col] = {
                'mean': np.mean(error_data),
                'std': np.std(error_data),
                'max': np.max(np.abs(error_data)),
                'rms': np.sqrt(np.mean(error_data**2))
            }
        
        return stats
    
    def plot_position_comparison(self, save_path=None):
        """
        绘制位置对比图（xyz）
        """
        if self.integrated_data is None:
            print("❌ 错误：请先进行速度积分")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Position Comparison: Velocity Integration vs Actual (XYZ)", 
                     fontsize=14, fontweight="bold", y=0.95)
        
        # X位置对比
        ax1 = axes[0]
        ax1.plot(self.integrated_data['timestamp'], self.integrated_data['x_actual'], 
                'b-', label='Actual (Filtered)', alpha=0.8, linewidth=2.0)
        ax1.plot(self.integrated_data['timestamp'], self.integrated_data['x_integrated'], 
                'r--', label='Integrated from Velocity', alpha=0.8, linewidth=1.5)
        ax1.set_ylabel('X Position (m)', fontsize=11)
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.set_title('X Position Comparison')
        
        # Y位置对比
        ax2 = axes[1]
        ax2.plot(self.integrated_data['timestamp'], self.integrated_data['y_actual'], 
                'b-', label='Actual (Filtered)', alpha=0.8, linewidth=2.0)
        ax2.plot(self.integrated_data['timestamp'], self.integrated_data['y_integrated'], 
                'r--', label='Integrated from Velocity', alpha=0.8, linewidth=1.5)
        ax2.set_ylabel('Y Position (m)', fontsize=11)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.set_title('Y Position Comparison')
        
        # Z位置对比
        ax3 = axes[2]
        ax3.plot(self.integrated_data['timestamp'], self.integrated_data['z_actual'], 
                'b-', label='Actual (Filtered)', alpha=0.8, linewidth=2.0)
        ax3.plot(self.integrated_data['timestamp'], self.integrated_data['z_integrated'], 
                'r--', label='Integrated from Velocity', alpha=0.8, linewidth=1.5)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Z Position (m)', fontsize=11)
        ax3.legend(fontsize=10, loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.set_title('Z Position Comparison')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 XYZ位置对比图已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)
    
    def plot_orientation_comparison(self, save_path=None):
        """
        绘制姿态对比图（RPY）
        """
        if self.integrated_data is None:
            print("❌ 错误：请先进行速度积分")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Orientation Comparison: Angular Velocity Integration vs Actual (RPY)", 
                     fontsize=14, fontweight="bold", y=0.95)
        
        # Roll对比
        ax1 = axes[0]
        ax1.plot(self.integrated_data['timestamp'], self.integrated_data['rx_actual'], 
                'b-', label='Actual (Filtered)', alpha=0.8, linewidth=2.0)
        ax1.plot(self.integrated_data['timestamp'], self.integrated_data['rx_integrated'], 
                'r--', label='Integrated from Angular Velocity', alpha=0.8, linewidth=1.5)
        ax1.set_ylabel('Roll (rad)', fontsize=11)
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.set_title('Roll Angle Comparison')
        
        # Pitch对比
        ax2 = axes[1]
        ax2.plot(self.integrated_data['timestamp'], self.integrated_data['ry_actual'], 
                'b-', label='Actual (Filtered)', alpha=0.8, linewidth=2.0)
        ax2.plot(self.integrated_data['timestamp'], self.integrated_data['ry_integrated'], 
                'r--', label='Integrated from Angular Velocity', alpha=0.8, linewidth=1.5)
        ax2.set_ylabel('Pitch (rad)', fontsize=11)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.set_title('Pitch Angle Comparison')
        
        # Yaw对比
        ax3 = axes[2]
        ax3.plot(self.integrated_data['timestamp'], self.integrated_data['rz_actual'], 
                'b-', label='Actual (Filtered)', alpha=0.8, linewidth=2.0)
        ax3.plot(self.integrated_data['timestamp'], self.integrated_data['rz_integrated'], 
                'r--', label='Integrated from Angular Velocity', alpha=0.8, linewidth=1.5)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Yaw (rad)', fontsize=11)
        ax3.legend(fontsize=10, loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.set_title('Yaw Angle Comparison')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 RPY姿态对比图已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)
    
    def plot_error_analysis(self, save_path=None):
        """
        绘制积分误差分析图
        """
        if self.integrated_data is None:
            print("❌ 错误：请先进行速度积分")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Integration Error Analysis", fontsize=14, fontweight="bold", y=0.95)
        
        # 位置误差
        error_configs = [
            ('x_error', 'X Position Error (m)', axes[0, 0]),
            ('y_error', 'Y Position Error (m)', axes[0, 1]),
            ('z_error', 'Z Position Error (m)', axes[0, 2]),
            ('rx_error', 'Roll Error (rad)', axes[1, 0]),
            ('ry_error', 'Pitch Error (rad)', axes[1, 1]),
            ('rz_error', 'Yaw Error (rad)', axes[1, 2])
        ]
        
        for error_col, title, ax in error_configs:
            error_data = self.integrated_data[error_col]
            ax.plot(self.integrated_data['timestamp'], error_data, 'r-', alpha=0.8, linewidth=1.5)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_title(f'{title}\nRMS: {np.sqrt(np.mean(error_data**2)):.6f}')
        
        # 只在底部子图添加x轴标签
        for ax in axes[1, :]:
            ax.set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 误差分析图已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)
    
    def save_integrated_data(self, save_path):
        """保存积分结果数据"""
        if self.integrated_data is None:
            print("❌ 错误：请先进行速度积分")
            return False
        
        try:
            self.integrated_data.to_csv(save_path, index=False)
            print(f"💾 积分结果已保存: {save_path}")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
    
    def run_analysis(self, smoothed_file, output_dir, initial_position=None):
        """
        运行完整的分析流程
        """
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. 加载数据
        if not self.load_smoothed_data(smoothed_file):
            return False
        
        # 2. 进行速度积分
        if not self.integrate_velocity_to_position(initial_position):
            return False
        
        # 3. 计算统计信息
        stats = self.calculate_statistics()
        if stats:
            print("\n📊 积分误差统计:")
            for var, stat in stats.items():
                print(f"  {var}: RMS={stat['rms']:.6f}, Max={stat['max']:.6f}, Std={stat['std']:.6f}")
        
        # 4. 生成对比图
        self.plot_position_comparison(Path(output_dir) / "position_comparison_xyz.png")
        self.plot_orientation_comparison(Path(output_dir) / "orientation_comparison_rpy.png")
        self.plot_error_analysis(Path(output_dir) / "integration_error_analysis.png")
        
        # 5. 保存积分结果
        # self.save_integrated_data(Path(output_dir) / "velocity_integrated_results.csv")
        
        print("\n🎉 速度积分分析完成！")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从速度反推位置并进行对比分析")
    parser.add_argument('--smoothed', type=str,
                        default='./output/smoothed_trajectory.csv',
                        help='滤波后的轨迹数据文件路径（18列格式）')
    parser.add_argument('--output_dir', type=str,
                        default='./plots',
                        help='输出图表和结果的目录')
    parser.add_argument('--initial_position', type=float, nargs=6,
                        help='初始位置 [x0 y0 z0 rx0 ry0 rz0]，如果不指定则使用数据第一个点')
    
    args = parser.parse_args()
    
    analyzer = VelocityToPositionAnalyzer()
    analyzer.run_analysis(args.smoothed, args.output_dir, args.initial_position)