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
plt.switch_backend('Agg')
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

    def calculate_kinematics_from_position(self, data, dt=1/60.0):
        """
        从位置数据通过差分计算速度和加速度
        :param data: 包含x,y,z位置的DataFrame
        :param dt: 时间步长（秒）
        :return: 包含计算出的速度和加速度的DataFrame副本
        """
        data_copy = data.copy()
        
        # 计算速度（位置的一阶差分）
        data_copy['vx_diff'] = np.gradient(data['x'], dt)
        data_copy['vy_diff'] = np.gradient(data['y'], dt)
        data_copy['vz_diff'] = np.gradient(data['z'], dt)
        
        # 计算加速度（速度的一阶差分）
        data_copy['ax_diff'] = np.gradient(data_copy['vx_diff'], dt)
        data_copy['ay_diff'] = np.gradient(data_copy['vy_diff'], dt)
        data_copy['az_diff'] = np.gradient(data_copy['vz_diff'], dt)
        
        return data_copy

    def integrate_velocity_to_position(self, data, dt=1/60.0, initial_position=None):
        """
        从速度数据通过积分计算位置
        :param data: 包含vx,vy,vz速度的DataFrame
        :param dt: 时间步长（秒）
        :param initial_position: 初始位置 [x0, y0, z0]，如果为None，则从data中获取第一个位置
        :return: 包含计算出的位置的DataFrame副本
        """
        data_copy = data.copy()

        if initial_position is None:
            # 尝试从数据中获取初始位置，如果不存在则默认为0
            x0 = data['x'].iloc[0] if 'x' in data.columns else 0.0
            y0 = data['y'].iloc[0] if 'y' in data.columns else 0.0
            z0 = data['z'].iloc[0] if 'z' in data.columns else 0.0
        else:
            x0, y0, z0 = initial_position

        # 积分速度得到位置
        data_copy['x_int'] = x0 + (data['vx'].cumsum() - data['vx'].iloc[0]) * dt
        data_copy['y_int'] = y0 + (data['vy'].cumsum() - data['vy'].iloc[0]) * dt
        data_copy['z_int'] = z0 + (data['vz'].cumsum() - data['vz'].iloc[0]) * dt

        return data_copy

    def load_data(self, original_file=None, smoothed_file=None, interpolated_file=None):
        """
        加载数据：支持两种格式
        1. 原始数据：6列 (x,y,z,rx,ry,rz) - 仅位置和姿态
        2. 滤波数据：18列 (x,y,z,vx,vy,vz,ax,ay,az,rx,ry,rz,vrx,vry,vrz,arx,ary,arz) - 完整状态
        3. 插值数据：18列 (x,y,z,vx,vy,vz,ax,ay,az,rx,ry,rz,vrx,vry,vrz,arx,ary,arz) - 完整状态
        """
        def load_original_data_core(filepath):
            """加载原始数据（6列格式）"""
            # 尝试读取CSV，可能有header
            try:
                df = pd.read_csv(filepath)
                if df.shape[1] >= 6:
                    # 有header的情况，重命名列
                    df = df.iloc[:, :6]
                    df.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
                else:
                    raise ValueError("Column count insufficient")
            except:
                # 无header的情况
                df = pd.read_csv(filepath, header=None)
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(subset=[0, 1, 2])
                
                if df.shape[1] < 6:
                    raise ValueError(f"Original data file {filepath} needs at least 6 columns (x,y,z,rx,ry,rz), got {df.shape[1]}")
                
                df = df.iloc[:, :6]
                df.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            
            # 添加计算得到的速度和加速度列（初始化为0）
            df['vx'] = 0.0
            df['vy'] = 0.0
            df['vz'] = 0.0
            df['ax'] = 0.0
            df['ay'] = 0.0
            df['az'] = 0.0
            df['vrx'] = 0.0
            df['vry'] = 0.0
            df['vrz'] = 0.0
            df['arx'] = 0.0
            df['ary'] = 0.0
            df['arz'] = 0.0
            
            return df

        def load_smoothed_data_core(filepath):
            """加载滤波数据（18列格式）"""
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
            
            return df

        def load_interpolated_data_core(filepath):
            """加载插值数据（6列格式，仅位置和姿态）"""
            try:
                df = pd.read_csv(filepath)
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                    df.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
                else:
                    raise ValueError("Column count insufficient")
            except:
                df = pd.read_csv(filepath, header=None)
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(subset=[0, 1, 2])
                
                if df.shape[1] < 6:
                    raise ValueError(f"Interpolated data file {filepath} needs at least 6 columns (x,y,z,rx,ry,rz), got {df.shape[1]}")
                
                df = df.iloc[:, :6]
                df.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            
            # 初始化速度和加速度列
            df['vx'] = 0.0
            df['vy'] = 0.0
            df['vz'] = 0.0
            df['ax'] = 0.0
            df['ay'] = 0.0
            df['az'] = 0.0
            df['vrx'] = 0.0
            df['vry'] = 0.0
            df['vrz'] = 0.0
            df['arx'] = 0.0
            df['ary'] = 0.0
            df['arz'] = 0.0
            
            return df

        try:
            # 1. 加载原始数据（如果提供）
            if original_file is not None:
                self.original_data = load_original_data_core(original_file)
                self.original_data['timestamp'] = np.arange(len(self.original_data)) / 60.0 # 60Hz
                
                # 计算原始数据的速度和加速度（通过差分）
                dt_orig = 1.0/60.0
                # 位置的速度和加速度
                self.original_data['vx'] = np.gradient(self.original_data['x'], dt_orig)
                self.original_data['vy'] = np.gradient(self.original_data['y'], dt_orig)
                self.original_data['vz'] = np.gradient(self.original_data['z'], dt_orig)
                self.original_data['ax'] = np.gradient(self.original_data['vx'], dt_orig)
                self.original_data['ay'] = np.gradient(self.original_data['vy'], dt_orig)
                self.original_data['az'] = np.gradient(self.original_data['vz'], dt_orig)
                
                # 姿态的角速度和角加速度
                self.original_data['vrx'] = np.gradient(self.original_data['rx'], dt_orig)
                self.original_data['vry'] = np.gradient(self.original_data['ry'], dt_orig)
                self.original_data['vrz'] = np.gradient(self.original_data['rz'], dt_orig)
                self.original_data['arx'] = np.gradient(self.original_data['vrx'], dt_orig)
                self.original_data['ary'] = np.gradient(self.original_data['vry'], dt_orig)
                self.original_data['arz'] = np.gradient(self.original_data['vrz'], dt_orig)
                
                print(f"✅ Original data loaded | Data points: {len(self.original_data)} | Columns: x,y,z,rx,ry,rz + calculated velocities/accelerations")

            # 2. 加载滤波数据（如果提供）
            if smoothed_file is not None:
                self.smoothed_data = load_smoothed_data_core(smoothed_file)
                self.smoothed_data['timestamp'] = np.arange(len(self.smoothed_data)) / 60.0 # 60Hz
                
                # Calculate differential velocity and acceleration for smoothed data (from position and attitude)
                dt_smooth = 1.0/60.0
                self.smoothed_data['vx_diff'] = np.gradient(self.smoothed_data['x'], dt_smooth)
                self.smoothed_data['vy_diff'] = np.gradient(self.smoothed_data['y'], dt_smooth)
                self.smoothed_data['vz_diff'] = np.gradient(self.smoothed_data['z'], dt_smooth)
                self.smoothed_data['ax_diff'] = np.gradient(self.smoothed_data['vx_diff'], dt_smooth)
                self.smoothed_data['ay_diff'] = np.gradient(self.smoothed_data['vy_diff'], dt_smooth)
                self.smoothed_data['az_diff'] = np.gradient(self.smoothed_data['vz_diff'], dt_smooth)
                
                self.smoothed_data['vrx_diff'] = np.gradient(self.smoothed_data['rx'], dt_smooth)
                self.smoothed_data['vry_diff'] = np.gradient(self.smoothed_data['ry'], dt_smooth)
                self.smoothed_data['vrz_diff'] = np.gradient(self.smoothed_data['rz'], dt_smooth)
                self.smoothed_data['arx_diff'] = np.gradient(self.smoothed_data['vrx_diff'], dt_smooth)
                self.smoothed_data['ary_diff'] = np.gradient(self.smoothed_data['vry_diff'], dt_smooth)
                self.smoothed_data['arz_diff'] = np.gradient(self.smoothed_data['vrz_diff'], dt_smooth)

                print(f"✅ Smoothed data loaded | Data points: {len(self.smoothed_data)} | Columns: x,y,z,vx,vy,vz,ax,ay,az,rx,ry,rz,vrx,vry,vrz,arx,ary,arz + calculated differential kinematics")
            
            # 3. 加载插值数据（如果提供）
            if interpolated_file is not None:
                self.interpolated_data = load_interpolated_data_core(interpolated_file) # 插值数据现在是6列格式
                self.interpolated_data['timestamp'] = np.arange(len(self.interpolated_data)) / 1000.0 # 1000Hz
                
                # Calculate differential velocity and acceleration for interpolated data (from position and attitude)
                dt_interp = 1.0/1000.0
                self.interpolated_data['vx'] = np.gradient(self.interpolated_data['x'], dt_interp)
                self.interpolated_data['vy'] = np.gradient(self.interpolated_data['y'], dt_interp)
                self.interpolated_data['vz'] = np.gradient(self.interpolated_data['z'], dt_interp)
                self.interpolated_data['ax'] = np.gradient(self.interpolated_data['vx'], dt_interp)
                self.interpolated_data['ay'] = np.gradient(self.interpolated_data['vy'], dt_interp)
                self.interpolated_data['az'] = np.gradient(self.interpolated_data['vz'], dt_interp)
                
                self.interpolated_data['vrx'] = np.gradient(self.interpolated_data['rx'], dt_interp)
                self.interpolated_data['vry'] = np.gradient(self.interpolated_data['ry'], dt_interp)
                self.interpolated_data['vrz'] = np.gradient(self.interpolated_data['rz'], dt_interp)
                self.interpolated_data['arx'] = np.gradient(self.interpolated_data['vrx'], dt_interp)
                self.interpolated_data['ary'] = np.gradient(self.interpolated_data['vry'], dt_interp)
                self.interpolated_data['arz'] = np.gradient(self.interpolated_data['vrz'], dt_interp)

                print(f"✅ Interpolated data loaded | Data points: {len(self.interpolated_data)} | Columns: x,y,z,rx,ry,rz + calculated velocities/accelerations")

            return True

        except FileNotFoundError as e:
            print(f"❌ Error: File not found - {e.filename}")
            return False
        except ValueError as e:
            print(f"❌ Data loading error: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred during data loading: {e}")
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
        style1 = {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.5, "alpha": 0.9}
        style2 = {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2.5, "alpha": 0.9}
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
        # plt.show()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 3D轨迹图已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_kinematics_comparison(self, save_path=None):
        """
        Velocity and acceleration comparison: Original (differential) vs Kalman Filter output
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 24), sharex=True) # 4行1列，共享x轴
        fig.suptitle("Kinematics Comparison: Original vs Kalman Filter vs Interpolated", fontsize=14, fontweight="bold", y=0.95)

        # 1. Position velocity comparison
        ax1 = axes[0]
        if self.original_data is not None: 
            vel_orig = np.sqrt(self.original_data['vx']**2 + self.original_data['vy']**2 + self.original_data['vz']**2)
            ax1.plot(self.original_data['timestamp'], vel_orig, 'b-', label='Original (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.smoothed_data is not None: 
            vel_smooth = np.sqrt(self.smoothed_data['vx']**2 + self.smoothed_data['vy']**2 + self.smoothed_data['vz']**2)
            ax1.plot(self.smoothed_data['timestamp'], vel_smooth, 'r-', label='Smoothed (Kalman Filter)', alpha=0.8, linewidth=2.0)
            
            vel_smooth_diff = np.sqrt(self.smoothed_data['vx_diff']**2 + self.smoothed_data['vy_diff']**2 + self.smoothed_data['vz_diff']**2)
            ax1.plot(self.smoothed_data['timestamp'], vel_smooth_diff, 'g--', label='Smoothed (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.interpolated_data is not None:
            vel_interp = np.sqrt(self.interpolated_data['vx']**2 + self.interpolated_data['vy']**2 + self.interpolated_data['vz']**2)
            ax1.plot(self.interpolated_data['timestamp'], vel_interp, 'm-', label='Interpolated (Differential)', alpha=0.8, linewidth=2.0)
            
            # vel_interp_diff = np.sqrt(self.interpolated_data['vx_diff']**2 + self.interpolated_data['vy_diff']**2 + self.interpolated_data['vz_diff']**2)
            # ax1.plot(self.interpolated_data['timestamp'], vel_interp_diff, 'c--', label='Interpolated (Differential)', alpha=0.8, linewidth=1.5)

        ax1.set_ylabel('Linear Velocity Magnitude (m/s)', fontsize=11)
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.set_title('Linear Velocity Profile Comparison')

        # 2. Position acceleration comparison
        ax2 = axes[1]
        if self.original_data is not None: 
            acc_orig = np.sqrt(self.original_data['ax']**2 + self.original_data['ay']**2 + self.original_data['az']**2)
            ax2.plot(self.original_data['timestamp'], acc_orig, 'b-', label='Original (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.smoothed_data is not None: 
            acc_smooth = np.sqrt(self.smoothed_data['ax']**2 + self.smoothed_data['ay']**2 + self.smoothed_data['az']**2)
            ax2.plot(self.smoothed_data['timestamp'], acc_smooth, 'r-', label='Smoothed (Kalman Filter)', alpha=0.8, linewidth=2.0)
            
            acc_smooth_diff = np.sqrt(self.smoothed_data['ax_diff']**2 + self.smoothed_data['ay_diff']**2 + self.smoothed_data['az_diff']**2)
            ax2.plot(self.smoothed_data['timestamp'], acc_smooth_diff, 'g--', label='Smoothed (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.interpolated_data is not None:
            acc_interp = np.sqrt(self.interpolated_data['ax']**2 + self.interpolated_data['ay']**2 + self.interpolated_data['az']**2)
            ax2.plot(self.interpolated_data['timestamp'], acc_interp, 'm-', label='Interpolated (Differential)', alpha=0.8, linewidth=2.0)
            
            # acc_interp_diff = np.sqrt(self.interpolated_data['ax_diff']**2 + self.interpolated_data['ay_diff']**2 + self.interpolated_data['az_diff']**2)
            # ax2.plot(self.interpolated_data['timestamp'], acc_interp_diff, 'c--', label='Interpolated (Differential)', alpha=0.8, linewidth=1.5)

        ax2.set_ylabel('Linear Acceleration Magnitude (m/s²)', fontsize=11)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.set_title('Linear Acceleration Profile Comparison')

        # 3. Angular velocity comparison
        ax3 = axes[2]
        if self.original_data is not None: 
            avel_orig = np.sqrt(self.original_data['vrx']**2 + self.original_data['vry']**2 + self.original_data['vrz']**2)
            ax3.plot(self.original_data['timestamp'], avel_orig, 'b-', label='Original (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.smoothed_data is not None: 
            avel_smooth = np.sqrt(self.smoothed_data['vrx']**2 + self.smoothed_data['vry']**2 + self.smoothed_data['vrz']**2)
            ax3.plot(self.smoothed_data['timestamp'], avel_smooth, 'r-', label='Smoothed (Kalman Filter)', alpha=0.8, linewidth=2.0)
            
            avel_smooth_diff = np.sqrt(self.smoothed_data['vrx_diff']**2 + self.smoothed_data['vry_diff']**2 + self.smoothed_data['vrz_diff']**2)
            ax3.plot(self.smoothed_data['timestamp'], avel_smooth_diff, 'g--', label='Smoothed (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.interpolated_data is not None:
            avel_interp = np.sqrt(self.interpolated_data['vrx']**2 + self.interpolated_data['vry']**2 + self.interpolated_data['vrz']**2)
            ax3.plot(self.interpolated_data['timestamp'], avel_interp, 'm-', label='Interpolated (Differential)', alpha=0.8, linewidth=2.0)
            
            # avel_interp_diff = np.sqrt(self.interpolated_data['vrx_diff']**2 + self.interpolated_data['vry_diff']**2 + self.interpolated_data['vrz_diff']**2)
            # ax3.plot(self.interpolated_data['timestamp'], avel_interp_diff, 'c--', label='Interpolated (Differential)', alpha=0.8, linewidth=1.5)

        ax3.set_ylabel('Angular Velocity Magnitude (rad/s)', fontsize=11)
        ax3.legend(fontsize=10, loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.set_title('Angular Velocity Profile Comparison')

        # 4. Angular acceleration comparison
        ax4 = axes[3]
        if self.original_data is not None: 
            aacc_orig = np.sqrt(self.original_data['arx']**2 + self.original_data['ary']**2 + self.original_data['arz']**2)
            ax4.plot(self.original_data['timestamp'], aacc_orig, 'b-', label='Original (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.smoothed_data is not None: 
            aacc_smooth = np.sqrt(self.smoothed_data['arx']**2 + self.smoothed_data['ary']**2 + self.smoothed_data['arz']**2)
            ax4.plot(self.smoothed_data['timestamp'], aacc_smooth, 'r-', label='Smoothed (Kalman Filter)', alpha=0.8, linewidth=2.0)
            
            aacc_smooth_diff = np.sqrt(self.smoothed_data['arx_diff']**2 + self.smoothed_data['ary_diff']**2 + self.smoothed_data['az_diff']**2)
            ax4.plot(self.smoothed_data['timestamp'], aacc_smooth_diff, 'g--', label='Smoothed (Differential)', alpha=0.8, linewidth=1.5)
        
        if self.interpolated_data is not None:
            aacc_interp = np.sqrt(self.interpolated_data['arx']**2 + self.interpolated_data['ary']**2 + self.interpolated_data['arz']**2)
            ax4.plot(self.interpolated_data['timestamp'], aacc_interp, 'm-', label='Interpolated (Differential)', alpha=0.8, linewidth=2.0)
            
            # aacc_interp_diff = np.sqrt(self.interpolated_data['arx_diff']**2 + self.interpolated_data['ary_diff']**2 + self.interpolated_data['arz_diff']**2)
            # ax4.plot(self.interpolated_data['timestamp'], aacc_interp_diff, 'c--', label='Interpolated (Differential)', alpha=0.8, linewidth=1.5)

        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Angular Acceleration Magnitude (rad/s²)', fontsize=11)
        ax4.legend(fontsize=10, loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.3)
        ax4.set_title('Angular Acceleration Profile Comparison')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # 调整布局，避免标题重叠
        # plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 运动学对比图已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_rpy_comparison(self, save_path=None):
        """
        RPY姿态时间序列对比：原始 vs 滤波 vs 插值
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
        fig.suptitle("RPY Orientation Comparison: Original vs Smoothed vs Interpolated", fontsize=14, fontweight="bold", y=0.95)

        # Roll
        ax1 = axes[0]
        if self.original_data is not None:
            ax1.plot(self.original_data['timestamp'], self.original_data['rx'], 'b-', label='Original', alpha=0.8, linewidth=1.5)
        if self.smoothed_data is not None:
            ax1.plot(self.smoothed_data['timestamp'], self.smoothed_data['rx'], 'r-', label='Smoothed', alpha=0.8, linewidth=2.0)
        if self.interpolated_data is not None:
            ax1.plot(self.interpolated_data['timestamp'], self.interpolated_data['rx'], 'm-', label='Interpolated', alpha=0.8, linewidth=2.0)
        ax1.set_ylabel('Roll (rad)', fontsize=11)
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.set_title('Roll Angle Over Time')

        # Pitch
        ax2 = axes[1]
        if self.original_data is not None:
            ax2.plot(self.original_data['timestamp'], self.original_data['ry'], 'b-', label='Original', alpha=0.8, linewidth=1.5)
        if self.smoothed_data is not None:
            ax2.plot(self.smoothed_data['timestamp'], self.smoothed_data['ry'], 'r-', label='Smoothed', alpha=0.8, linewidth=2.0)
        if self.interpolated_data is not None:
            ax2.plot(self.interpolated_data['timestamp'], self.interpolated_data['ry'], 'm-', label='Interpolated', alpha=0.8, linewidth=2.0)
        ax2.set_ylabel('Pitch (rad)', fontsize=11)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.set_title('Pitch Angle Over Time')

        # Yaw
        ax3 = axes[2]
        if self.original_data is not None:
            ax3.plot(self.original_data['timestamp'], self.original_data['rz'], 'b-', label='Original', alpha=0.8, linewidth=1.5)
        if self.smoothed_data is not None:
            ax3.plot(self.smoothed_data['timestamp'], self.smoothed_data['rz'], 'r-', label='Smoothed', alpha=0.8, linewidth=2.0)
        if self.interpolated_data is not None:
            ax3.plot(self.interpolated_data['timestamp'], self.interpolated_data['rz'], 'm-', label='Interpolated', alpha=0.8, linewidth=2.0)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Yaw (rad)', fontsize=11)
        ax3.legend(fontsize=10, loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.set_title('Yaw Angle Over Time')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        # plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"📊 RPY姿态对比图已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def run_comparison(self, original_file, smoothed_file, interpolated_file, output_dir):
        """
        运行所有对比并保存图表
        """
        if not self.load_data(original_file, smoothed_file, interpolated_file):
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. 3D轨迹对比：原始 vs 滤波
        if self.original_data is not None and self.smoothed_data is not None:
            self.plot_3d_trajectory_pair(self.original_data, "Original", self.smoothed_data, "Smoothed",
                                         "3D Trajectory Comparison: Original vs Smoothed",
                                         Path(output_dir) / "3d_original_vs_smoothed.png")

        # 2. 3D轨迹对比：原始 vs 插值
        if self.original_data is not None and self.interpolated_data is not None:
            self.plot_3d_trajectory_pair(self.original_data, "Original", self.interpolated_data, "Interpolated",
                                         "3D Trajectory Comparison: Original vs Interpolated",
                                         Path(output_dir) / "3d_original_vs_interpolated.png")

        # 3. 运动学对比：原始 vs 滤波 vs 插值
        if self.original_data is not None or self.smoothed_data is not None or self.interpolated_data is not None:
            self.plot_kinematics_comparison(Path(output_dir) / "kinematics_comparison.png")

        # 4. RPY姿态时间序列对比：原始 vs 滤波 vs 插值
        if self.original_data is not None or self.smoothed_data is not None or self.interpolated_data is not None:
            self.plot_rpy_comparison(Path(output_dir) / "rpy_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 3D trajectories and RPY orientations.")
    parser.add_argument('--original', type=str,
                        default='./data/trajectory.csv',  # Default original data path
                        help='Path to the original trajectory CSV file (x,y,z,rx,ry,rz).')
    parser.add_argument('--smoothed', type=str,
                        default='./output/smoothed_trajectory.csv',  # Default smoothed data path
                        help='Path to the smoothed trajectory CSV file (18 columns).')
    parser.add_argument('--interpolated', type=str,
                        default='./output/interpolated_pose_1kHz.csv',  # Default interpolated data path
                        help='Path to the interpolated trajectory CSV file (6 or 18 columns).')
    parser.add_argument('--output_dir', type=str,
                        default='./plots',  # Default output directory for plots
                        help='Directory to save the comparison plots.')

    args = parser.parse_args()

    comparator = Trajectory3DComparator()
    comparator.run_comparison(args.original, args.smoothed, args.interpolated, args.output_dir)