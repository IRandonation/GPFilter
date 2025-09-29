#!/usr/bin/env python3
"""
滤波效果对比绘图工具
对比原始数据和滤波后数据的xyz位置和rpy姿态
生成多种对比图表：
1. 3D轨迹对比图
2. 位置时间序列对比图 (x, y, z)
3. 姿态时间序列对比图 (rx, ry, rz)
4. 统计分析图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# 设置matplotlib后端和样式
plt.switch_backend('Agg')
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.linewidth"] = 1.0
plt.style.use('default')


class FilterComparisonPlotter:
    def __init__(self):
        self.original_data = None
        self.filtered_data = None
        self.time_steps = None
        
    def load_data(self, original_file, filtered_file):
        """
        加载原始数据和滤波后数据
        
        Args:
            original_file: 原始数据文件路径
            filtered_file: 滤波后数据文件路径
        """
        try:
            # 尝试读取原始数据，如果没有列标题则添加
            try:
                self.original_data = pd.read_csv(original_file)
                # 检查是否有正确的列名
                if 'x' not in self.original_data.columns:
                    # 没有列标题，重新读取并添加列名
                    self.original_data = pd.read_csv(original_file, header=None, 
                                                   names=['x', 'y', 'z', 'rx', 'ry', 'rz'])
            except:
                self.original_data = pd.read_csv(original_file, header=None, 
                                               names=['x', 'y', 'z', 'rx', 'ry', 'rz'])
            
            self.filtered_data = pd.read_csv(filtered_file)
            
            print(f"原始数据形状: {self.original_data.shape}")
            print(f"滤波数据形状: {self.filtered_data.shape}")
            print(f"原始数据列: {list(self.original_data.columns)}")
            print(f"滤波数据列: {list(self.filtered_data.columns)}")
            
            # 确保数据长度一致
            min_length = min(len(self.original_data), len(self.filtered_data))
            self.original_data = self.original_data.iloc[:min_length]
            self.filtered_data = self.filtered_data.iloc[:min_length]
            
            # 生成时间步
            self.time_steps = np.arange(len(self.original_data))
            
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def plot_3d_trajectory_comparison(self, save_path=None):
        """
        绘制3D轨迹对比图
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原始轨迹
        ax.plot(self.original_data['x'], self.original_data['y'], self.original_data['z'],
                'r-', alpha=0.7, linewidth=1.5, label='Original Trajectory')
        
        # 绘制滤波后轨迹
        ax.plot(self.filtered_data['x'], self.filtered_data['y'], self.filtered_data['z'],
                'b-', alpha=0.8, linewidth=2, label='Filtered Trajectory')
        
        # 标记起点和终点
        ax.scatter(self.original_data['x'].iloc[0], self.original_data['y'].iloc[0], 
                  self.original_data['z'].iloc[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(self.original_data['x'].iloc[-1], self.original_data['y'].iloc[-1], 
                  self.original_data['z'].iloc[-1], c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory Comparison: Original vs Filtered', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D轨迹对比图已保存: {save_path}")
        
        plt.close()
    
    def plot_position_time_series(self, save_path=None):
        """
        绘制位置时间序列对比图
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        positions = ['x', 'y', 'z']
        colors_orig = ['red', 'green', 'blue']
        colors_filt = ['darkred', 'darkgreen', 'darkblue']
        
        for i, pos in enumerate(positions):
            # 原始数据
            axes[i].plot(self.time_steps, self.original_data[pos], 
                        color=colors_orig[i], alpha=0.6, linewidth=1, 
                        label=f'Original {pos.upper()}')
            
            # 滤波数据
            axes[i].plot(self.time_steps, self.filtered_data[pos], 
                        color=colors_filt[i], linewidth=2, 
                        label=f'Filtered {pos.upper()}')
            
            axes[i].set_ylabel(f'{pos.upper()} (m)')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.3)
            axes[i].set_title(f'{pos.upper()} Axis Position Comparison')
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('Position Time Series Comparison: Original vs Filtered', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"位置时间序列对比图已保存: {save_path}")
        
        plt.close()
    
    def plot_orientation_time_series(self, save_path=None):
        """
        绘制姿态时间序列对比图
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        orientations = ['rx', 'ry', 'rz']
        labels = ['Roll', 'Pitch', 'Yaw']
        colors_orig = ['orange', 'purple', 'brown']
        colors_filt = ['darkorange', 'indigo', 'maroon']
        
        for i, (orient, label) in enumerate(zip(orientations, labels)):
            # 原始数据
            axes[i].plot(self.time_steps, self.original_data[orient], 
                        color=colors_orig[i], alpha=0.6, linewidth=1, 
                        label=f'Original {label}')
            
            # 滤波数据
            axes[i].plot(self.time_steps, self.filtered_data[orient], 
                        color=colors_filt[i], linewidth=2, 
                        label=f'Filtered {label}')
            
            axes[i].set_ylabel(f'{label} (rad)')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.3)
            axes[i].set_title(f'{label} Angle Comparison')
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('Orientation Time Series Comparison: Original vs Filtered', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"姿态时间序列对比图已保存: {save_path}")
        
        plt.close()
    
    def plot_error_analysis(self, save_path=None):
        """
        绘制误差分析图
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 计算位置误差
        pos_errors = {}
        for pos in ['x', 'y', 'z']:
            pos_errors[pos] = self.original_data[pos] - self.filtered_data[pos]
        
        # 计算姿态误差
        orient_errors = {}
        for orient in ['rx', 'ry', 'rz']:
            orient_errors[orient] = self.original_data[orient] - self.filtered_data[orient]
        
        # 绘制位置误差
        for i, (pos, error) in enumerate(pos_errors.items()):
            axes[0, i].plot(self.time_steps, error, 'r-', alpha=0.7)
            axes[0, i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, i].set_title(f'{pos.upper()} Axis Error')
            axes[0, i].set_ylabel('Error (m)')
            axes[0, i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_error = np.mean(np.abs(error))
            std_error = np.std(error)
            axes[0, i].text(0.02, 0.98, f'Mean Absolute Error: {mean_error:.6f}m\nStandard Deviation: {std_error:.6f}m',
                           transform=axes[0, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 绘制姿态误差
        labels = ['Roll', 'Pitch', 'Yaw']
        for i, (orient, error, label) in enumerate(zip(['rx', 'ry', 'rz'], 
                                                      [orient_errors['rx'], orient_errors['ry'], orient_errors['rz']], 
                                                      labels)):
            axes[1, i].plot(self.time_steps, error, 'b-', alpha=0.7)
            axes[1, i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, i].set_title(f'{label} Error')
            axes[1, i].set_ylabel('Error (rad)')
            axes[1, i].set_xlabel('Time Step')
            axes[1, i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_error = np.mean(np.abs(error))
            std_error = np.std(error)
            axes[1, i].text(0.02, 0.98, f'Mean Absolute Error: {mean_error:.6f}rad\nStandard Deviation: {std_error:.6f}rad',
                           transform=axes[1, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Filtering Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"误差分析图已保存: {save_path}")
        
        plt.close()
    
    def plot_statistics_comparison(self, save_path=None):
        """
        绘制统计对比图
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 计算统计量
        variables = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        orig_means = [self.original_data[var].mean() for var in variables]
        filt_means = [self.filtered_data[var].mean() for var in variables]
        orig_stds = [self.original_data[var].std() for var in variables]
        filt_stds = [self.filtered_data[var].std() for var in variables]
        
        x_pos = np.arange(len(variables))
        
        # Mean Comparison
        width = 0.35
        axes[0, 0].bar(x_pos - width/2, orig_means, width, label='Original Data', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, filt_means, width, label='Filtered Data', alpha=0.7)
        axes[0, 0].set_xlabel('Variable')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].set_title('Mean Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(variables)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard Deviation Comparison
        axes[0, 1].bar(x_pos - width/2, orig_stds, width, label='Original Data', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, filt_stds, width, label='Filtered Data', alpha=0.7)
        axes[0, 1].set_xlabel('Variable')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Standard Deviation Comparison (Noise Level)')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(variables)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Position Data Distribution Comparison
        pos_data = pd.DataFrame({
            'Original X': self.original_data['x'],
            'Filtered X': self.filtered_data['x'],
            'Original Y': self.original_data['y'],
            'Filtered Y': self.filtered_data['y'],
            'Original Z': self.original_data['z'],
            'Filtered Z': self.filtered_data['z']
        })
        pos_data.boxplot(ax=axes[1, 0])
        axes[1, 0].set_title('Position Data Distribution Comparison')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Orientation Data Distribution Comparison
        orient_data = pd.DataFrame({
            'Original RX': self.original_data['rx'],
            'Filtered RX': self.filtered_data['rx'],
            'Original RY': self.original_data['ry'],
            'Filtered RY': self.filtered_data['ry'],
            'Original RZ': self.original_data['rz'],
            'Filtered RZ': self.filtered_data['rz']
        })
        orient_data.boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Orientation Data Distribution Comparison')
        axes[1, 1].set_ylabel('Angle (rad)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Statistical Characteristics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"统计对比图已保存: {save_path}")
        
        plt.close()
    
    def generate_all_plots(self, output_dir='../plots'):
        """
        生成所有对比图
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating filter comparison plots...")
        
        # Generate various comparison plots
        self.plot_3d_trajectory_comparison(output_path / 'filter_3d_trajectory_comparison.png')
        self.plot_position_time_series(output_path / 'filter_position_time_series.png')
        self.plot_orientation_time_series(output_path / 'filter_orientation_time_series.png')
        self.plot_error_analysis(output_path / 'filter_error_analysis.png')
        self.plot_statistics_comparison(output_path / 'filter_statistics_comparison.png')
        
        print(f"All plots saved to: {output_path}")
    
    def print_summary_statistics(self):
        """
        打印汇总统计信息
        """
        print("\n=== Filter Effect Summary Statistics ===")
        
        variables = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        
        for var in variables:
            orig_std = self.original_data[var].std()
            filt_std = self.filtered_data[var].std()
            noise_reduction = (orig_std - filt_std) / orig_std * 100
            
            print(f"{var.upper()}:")
            print(f"  Original Std Dev: {orig_std:.6f}")
            print(f"  Filtered Std Dev: {filt_std:.6f}")
            print(f"  Noise Reduction: {noise_reduction:.2f}%")
            print()


def main():
    parser = argparse.ArgumentParser(description='Filtering Effect Comparison Plotting Tool')
    parser.add_argument('--original', '-o',
                       default='./data/trajectory.csv',
                       help='Path to the original data file (default: ../data/trajectory.csv)')
    parser.add_argument('--filtered', '-f',
                       default='./output/smoothed_trajectory.csv',
                       help='Path to the filtered data file (default: ../output/smoothed_trajectory.csv)')
    parser.add_argument('--output-dir', '-d',
                       default='./plots',
                       help='Output directory for plots (default: ./plots)')
    parser.add_argument('--stats', action='store_true',
                       help='Display detailed statistics')
    
    args = parser.parse_args()
    
    # Check input files
    if not Path(args.original).exists():
        print(f"Error: Original data file not found: {args.original}")
        return 1
    
    if not Path(args.filtered).exists():
        print(f"Error: Filtered data file not found: {args.filtered}")
        return 1
    
    # Create plotter and load data
    plotter = FilterComparisonPlotter()
    if not plotter.load_data(args.original, args.filtered):
        return 1
    
    # Generate all plots
    plotter.generate_all_plots(args.output_dir)
    
    # Display statistics
    if args.stats:
        plotter.print_summary_statistics()
    
    print("Filtering effect comparison analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())