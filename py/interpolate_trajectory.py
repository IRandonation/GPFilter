#!/usr/bin/env python3
"""
Trajectory Interpolation Script (Pose-Only CubicSpline Version)

This script interpolates 6-DoF pose (x,y,z,rx,ry,rz) from 60 Hz to 1 kHz
using CubicSpline interpolation.  No velocities/accelerations are computed
or saved.

Usage:
    python interpolate_trajectory.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--target-freq TARGET_FREQ]

Default:
    - Input : ../output/smoothed_trajectory.csv
    - Output: ../output/interpolated_pose_1kHz.csv
    - Target frequency: 1000 Hz
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys

from scipy.interpolate import CubicSpline

POSE_COLS = ['x', 'y', 'z', 'rx', 'ry', 'rz']

# ---------- 核心逻辑 ----------
def interpolate_pose(input_file: str, output_file: str,
                     original_freq: float = 60.0,
                     target_freq: float = 1000.0) -> bool:
    try:
        print(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        if df.empty:
            print("Error: Input file is empty")
            return False

        # 只保留位姿列
        if not all(c in df.columns for c in POSE_COLS):
            print("Error: Input CSV must contain x,y,z,rx,ry,rz")
            return False
        df_pose = df[POSE_COLS]

        # 时间向量
        t_old = np.arange(len(df_pose)) / original_freq
        t_max = t_old[-1]
        t_new = np.arange(0, t_max + 1/target_freq, 1/target_freq)
        t_new = t_new[t_new <= t_max]          # 防浮点越界

        print(f"Original points: {len(t_old)} → Interpolated points: {len(t_new)}")

        # CubicSpline 插值
        splines = {c: CubicSpline(t_old, df_pose[c].values) for c in POSE_COLS}
        interp_dict = {c: splines[c](t_new) for c in POSE_COLS}

        # 保存
        out_dir = os.path.dirname(output_file)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(interp_dict).to_csv(output_file, index=False)
        print(f"Pose-only interpolated data saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

# ---------- 命令行入口 ----------
def main():
    parser = argparse.ArgumentParser(description="Interpolate 6-DoF pose to 1 kHz")
    parser.add_argument('--input', '-i',
                        default='../output/smoothed_trajectory.csv',
                        help='Input CSV (must contain x,y,z,rx,ry,rz)')
    parser.add_argument('--output', '-o',
                        default='../output/interpolated_pose_1kHz.csv',
                        help='Output CSV with interpolated pose only')
    parser.add_argument('--original-freq', type=float, default=60.0)
    parser.add_argument('--target-freq', type=float, default=1000.0)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, args.input)
    output_file = os.path.join(script_dir, args.output)

    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        return 1

    ok = interpolate_pose(input_file, output_file,
                          args.original_freq, args.target_freq)
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())