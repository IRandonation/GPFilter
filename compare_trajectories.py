#!/usr/bin/env python3
"""
Trajectory Data Comparison Tool
用于对比原始轨迹数据与滤波、插值轨迹数据
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


class TrajectoryComparator:
    """轨迹数据对比器"""
    
    def __init__(self):
        self.original_data = None
        self.smoothed_data = None
        self.interpolated_data = None
        
    def load_data(self, original_file, smoothed_file, interpolated_file):
        """加载数据文件并正确映射列名"""
        try:
            # 加载原始轨迹数据 (x, y)
            self.original_data = pd.read_csv(original_file, header=None,
                                           names=['x', 'y'])
            self.original_data = self.original_data.apply(pd.to_numeric, errors='coerce')
            print(f"成功加载原始数据: {original_file}")
            print(f"数据点数量: {len(self.original_data)}")
            
            # 加载滤波轨迹数据 (x, y)
            self.smoothed_data = pd.read_csv(smoothed_file, header=None,
                                           names=['x_smooth', 'y_smooth'])
            self.smoothed_data = self.smoothed_data.apply(pd.to_numeric, errors='coerce')
            print(f"成功加载滤波数据: {smoothed_file}")
            print(f"数据点数量: {len(self.smoothed_data)}")
            
            # 加载插值轨迹数据（带时间戳、速度和加速度）
            if interpolated_file:
                self.interpolated_data = pd.read_csv(interpolated_file)
                self.interpolated_data = self.interpolated_data.apply(pd.to_numeric, errors='coerce')
                print(f"成功加载插值数据: {interpolated_file}")
                print(f"数据点数量: {len(self.interpolated_data)}")
                print(f"时间范围: {self.interpolated_data['timestamp'].min():.3f} 到 {self.interpolated_data['timestamp'].max():.3f} 秒")
            
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def calculate_statistics(self):
        """计算原始轨迹与滤波轨迹之间的统计信息"""
        if self.original_data is None or self.smoothed_data is None:
            print("错误: 请先加载数据")
            return None
            
        # 清除NaN值
        orig_clean = self.original_data.dropna()
        smooth_clean = self.smoothed_data.dropna()
        
        # 确保数据长度一致
        min_length = min(len(orig_clean), len(smooth_clean))
        orig_x = orig_clean['x'].values[:min_length]
        orig_y = orig_clean['y'].values[:min_length]
        smooth_x = smooth_clean['x_smooth'].values[:min_length]
        smooth_y = smooth_clean['y_smooth'].values[:min_length]
        
        # 计算误差
        error_x = orig_x - smooth_x
        error_y = orig_y - smooth_y
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        # 移除误差中的NaN值
        error_x = error_x[~np.isnan(error_x)]
        error_y = error_y[~np.isnan(error_y)]
        error_magnitude = error_magnitude[~np.isnan(error_magnitude)]
        
        if len(error_magnitude) == 0:
            print("警告: 没有有效的数据点用于误差计算")
            return None
        
        stats = {
            'x_error_mean': np.mean(error_x),
            'x_error_std': np.std(error_x),
            'y_error_mean': np.mean(error_y),
            'y_error_std': np.std(error_y),
            'magnitude_error_mean': np.mean(error_magnitude),
            'magnitude_error_std': np.std(error_magnitude),
            'max_error': np.max(error_magnitude),
            'min_error': np.min(error_magnitude)
        }
        
        return stats
    
    def print_statistics(self, stats):
        """以英文表格形式打印统计信息"""
        if stats is None:
            return
            
        print("\n=== Error Statistics ===")
        print(f"X Direction Error - Mean: {stats['x_error_mean']:.6f}, Std: {stats['x_error_std']:.6f}")
        print(f"Y Direction Error - Mean: {stats['y_error_mean']:.6f}, Std: {stats['y_error_std']:.6f}")
        print(f"Magnitude Error - Mean: {stats['magnitude_error_mean']:.6f}, Std: {stats['magnitude_error_std']:.6f}")
        print(f"Max Error: {stats['max_error']:.6f}")
        print(f"Min Error: {stats['min_error']:.6f}")
    
    def plot_comparison(self, save_path=None):
        """生成所有需要的对比图表"""
        if self.original_data is None or self.smoothed_data is None:
            print("错误: 请先加载数据")
            return
            
        # 检查是否有插值数据
        has_interpolated_data = self.interpolated_data is not None
        
        # 创建图形和子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('trajectory compare', fontsize=16)
        
        # 清除NaN值
        orig_clean = self.original_data.dropna()
        smooth_clean = self.smoothed_data.dropna()
        
        # 确保数据长度一致
        min_length = min(len(orig_clean), len(smooth_clean))
        
        # 1. 原轨迹与滤波轨迹对比
        ax1 = axes[0, 0]
        ax1.plot(orig_clean['x'][:min_length],
                orig_clean['y'][:min_length],
                'b-', label='Original', alpha=0.7, linewidth=2)
        ax1.plot(smooth_clean['x_smooth'][:min_length],
                smooth_clean['y_smooth'][:min_length],
                'r-', label='Smoothed', alpha=0.7, linewidth=2)
        
        # 添加标记点以便更清晰地看到对应关系
        ax1.scatter(orig_clean['x'][:min_length], orig_clean['y'][:min_length], 
                   c='blue', s=30, alpha=0.5)
        ax1.scatter(smooth_clean['x_smooth'][:min_length], smooth_clean['y_smooth'][:min_length], 
                   c='red', s=30, alpha=0.5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Original vs Smoothed Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 滤波轨迹与插值轨迹对比（如果有插值数据）
        ax2 = axes[0, 1]
        if has_interpolated_data:
            interp_clean = self.interpolated_data.dropna()
            
            ax2.plot(smooth_clean['x_smooth'][:min_length],
                    smooth_clean['y_smooth'][:min_length],
                    'r-', label='Smoothed', alpha=0.7, linewidth=2)
            ax2.scatter(smooth_clean['x_smooth'][:min_length], 
                       smooth_clean['y_smooth'][:min_length], 
                       c='red', s=30, alpha=0.5)
            
            # 绘制插值轨迹
            ax2.plot(interp_clean['x'],
                    interp_clean['y'],
                    'b--', label='GP Interpolated', alpha=0.7, linewidth=1.85)
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Smoothed vs GP Interpolated Trajectory')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No interpolation data', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Smoothed vs GP Interpolated Trajectory')
        
        # 3. 速度曲线（如果有插值数据）
        ax3 = axes[1, 0]
        if has_interpolated_data:
            interp_clean = self.interpolated_data.dropna()
            
            ax3.plot(interp_clean['timestamp'],
                    interp_clean['velocity'],
                    'b-', alpha=0.7, linewidth=2)
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Velocity (m/s)')
            ax3.set_title('GP Interpolated Velocity Profile')
            ax3.grid(True, alpha=0.3)
            
            # 优化坐标轴范围
            y_min, y_max = interp_clean['velocity'].min(), interp_clean['velocity'].max()
            ax3.set_ylim(max(0, y_min - 0.1), y_max + 0.1)
        else:
            ax3.text(0.5, 0.5, 'No interpolation data', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12)
            ax3.set_title('GP Interpolated Velocity Profile')
        
        # 4. 加速度曲线（如果有插值数据）
        ax4 = axes[1, 1]
        if has_interpolated_data:
            interp_clean = self.interpolated_data.dropna()
            
            ax4.plot(interp_clean['timestamp'],
                    interp_clean['acceleration'],
                    'orange', alpha=0.7, linewidth=2)
            
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Acceleration (m/s²)')
            ax4.set_title('GP Interpolated Acceleration Profile')
            ax4.grid(True, alpha=0.3)
            
            # 优化坐标轴范围
            y_min, y_max = interp_clean['acceleration'].min(), interp_clean['acceleration'].max()
            ax4.set_ylim(max(0, y_min - 0.1), y_max + 0.1)
        else:
            ax4.text(0.5, 0.5, 'No interpolation data', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12)
            ax4.set_title('GP Interpolated Acceleration Profile')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
        
        # 保存图像
        if not save_path:
            save_path = 'trajectory_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
        print("注意: 图表已保存为图像文件，请查看生成的图像")
    
    def run_comparison(self, original_file, smoothed_file, interpolated_file=None, save_plot=None):
        """运行完整的对比分析"""
        print("开始轨迹数据对比分析...")
        
        # 加载数据
        if not self.load_data(original_file, smoothed_file, interpolated_file):
            return False
        
        # 计算统计信息
        stats = self.calculate_statistics()
        self.print_statistics(stats)
        
        # 绘制对比图表
        self.plot_comparison(save_plot)
        
        print("对比分析完成!")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='轨迹数据对比工具')
    parser.add_argument('--original', '-o', default='data/trajectory.csv',
                       help='原始轨迹数据文件路径')
    parser.add_argument('--smoothed', '-s', default='output/smoothed_trajectory.csv',
                       help='滤波后轨迹数据文件路径')
    parser.add_argument('--interpolated', '-i', default='output/interpolated_trajectory.csv',
                       help='插值后轨迹数据文件路径')
    parser.add_argument('--save-plot', '-p',
                       help='图表保存路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for file_path in [args.original, args.smoothed]:
        if not Path(file_path).exists():
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
    
    if args.interpolated and not Path(args.interpolated).exists():
        print(f"警告: 插值数据文件不存在: {args.interpolated}")
        args.interpolated = None
    
    # 创建比较器并运行分析
    comparator = TrajectoryComparator()
    comparator.run_comparison(
        original_file=args.original,
        smoothed_file=args.smoothed,
        interpolated_file=args.interpolated,
        save_plot=args.save_plot
    )


if __name__ == "__main__":
    main()
