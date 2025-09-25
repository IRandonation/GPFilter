#!/usr/bin/env python3
"""
欧拉角转四元数转换工具
读取包含x,y,z,rx,ry,rz的CSV文件，将欧拉角(rx,ry,rz)转换为四元数(qw,qx,qy,qz)
输出新的CSV文件包含x,y,z,qw,qx,qy,qz
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def euler_to_quaternion(roll, pitch, yaw):
    """
    将欧拉角(roll, pitch, yaw)转换为四元数(qw, qx, qy, qz)
    
    Args:
        roll: 绕X轴旋转角度（弧度）
        pitch: 绕Y轴旋转角度（弧度）
        yaw: 绕Z轴旋转角度（弧度）
    
    Returns:
        tuple: (qw, qx, qy, qz) 四元数分量
    """
    # 计算半角
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # 计算四元数分量
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qw, qx, qy, qz


def convert_csv_euler_to_quaternion(input_file, output_file=None):
    """
    读取CSV文件，将欧拉角转换为四元数并保存
    
    Args:
        input_file: 输入CSV文件路径，支持6列格式(x,y,z,rx,ry,rz)或18列滤波格式
        output_file: 输出CSV文件路径，如果为None则自动生成waypoints.csv
    
    Returns:
        str: 输出文件路径
    """
    # 读取输入文件
    try:
        data = pd.read_csv(input_file)
        print(f"成功读取文件: {input_file}")
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    # 检查必要的列是否存在
    required_columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"缺少必要的列: {missing_columns}")
        return None
    
    # 创建输出数据框
    output_data = pd.DataFrame()
    output_data['x'] = data['x']
    output_data['y'] = data['y']
    output_data['z'] = data['z']
    
    # 转换欧拉角为四元数
    print("正在转换欧拉角为四元数...")
    quaternions = []
    
    for i, row in data.iterrows():
        roll = row['rx']
        pitch = row['ry']
        yaw = row['rz']
        
        qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
        quaternions.append([qw, qx, qy, qz])
    
    # 添加四元数列
    quaternions = np.array(quaternions)
    output_data['qw'] = quaternions[:, 0]
    output_data['qx'] = quaternions[:, 1]
    output_data['qy'] = quaternions[:, 2]
    output_data['qz'] = quaternions[:, 3]
    
    # 生成输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / "waypoints.csv"
    
    # 保存输出文件
    try:
        output_data.to_csv(output_file, index=False)
        print(f"成功保存文件: {output_file}")
        print(f"输出数据形状: {output_data.shape}")
        print(f"输出列名: {list(output_data.columns)}")
        
        # 显示前几行数据作为验证
        print("\n前5行转换结果:")
        print(output_data.head())
        
        return str(output_file)
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None


def verify_quaternion_conversion(input_file, output_file):
    """
    验证四元数转换的正确性
    
    Args:
        input_file: 原始欧拉角文件
        output_file: 转换后的四元数文件
    """
    print("\n=== 验证四元数转换 ===")
    
    # 读取原始数据和转换后数据
    original = pd.read_csv(input_file)
    converted = pd.read_csv(output_file)
    
    # 检查几个样本点
    sample_indices = [0, len(original)//2, len(original)-1]
    
    for i in sample_indices:
        print(f"\n样本 {i}:")
        print(f"  原始欧拉角: rx={original.iloc[i]['rx']:.6f}, ry={original.iloc[i]['ry']:.6f}, rz={original.iloc[i]['rz']:.6f}")
        
        qw, qx, qy, qz = converted.iloc[i]['qw'], converted.iloc[i]['qx'], converted.iloc[i]['qy'], converted.iloc[i]['qz']
        print(f"  转换四元数: qw={qw:.6f}, qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}")
        
        # 验证四元数模长应该为1
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        print(f"  四元数模长: {norm:.6f} (应该接近1.0)")
        
        if abs(norm - 1.0) > 1e-6:
            print(f"  警告: 四元数模长不为1!")


def main():
    parser = argparse.ArgumentParser(description='将滤波后的轨迹数据转换为waypoints (xyz + 四元数)')
    parser.add_argument('input_file', nargs='?', 
                       default='./output/smoothed_trajectory.csv',
                       help='输入CSV文件路径 (默认: ./output/smoothed_trajectory.csv)')
    parser.add_argument('-o', '--output', 
                       help='输出CSV文件路径 (默认: waypoints.csv)')
    parser.add_argument('--use-interpolated', action='store_true',
                       help='使用插值数据 (./output/interpolated_smoothed_trajectory.csv)')
    parser.add_argument('--verify', action='store_true',
                       help='验证转换结果')
    
    args = parser.parse_args()
    
    # 选择输入文件：如果指定使用插值数据，则覆盖默认输入文件
    input_file = args.input_file
    if args.use_interpolated:
        input_file = './output/interpolated_smoothed_trajectory.csv'
        print(f"使用插值数据: {input_file}")
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return 1
    
    # 执行转换
    output_file = convert_csv_euler_to_quaternion(input_file, args.output)
    
    if output_file is None:
        print("转换失败!")
        return 1
    
    # 验证转换结果
    if args.verify:
        verify_quaternion_conversion(input_file, output_file)
    
    print(f"\n转换完成! 输出文件: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())