#!/usr/bin/env python3
"""
Script to verify the effectiveness of the RTS smoother.
Compares raw data with smoothed data, checking for continuity in velocity and acceleration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """Loads raw and smoothed data."""
    # Load raw data
    raw_data = pd.read_csv('data/trajectory.csv', header=None)
    raw_data.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    
    # Load smoothed data
    smooth_data = pd.read_csv('output/smoothed_trajectory.csv')
    
    return raw_data, smooth_data

def calculate_derivatives(data, dt=1/60.0):
    """Calculates numerical derivatives."""
    pos_cols = ['x', 'y', 'z']
    vel_cols = ['vx', 'vy', 'vz']
    acc_cols = ['ax', 'ay', 'az']
    
    # Calculate velocity (derivative of position)
    for i, pos_col in enumerate(pos_cols):
        vel_col = vel_cols[i]
        data[vel_col + '_calc'] = np.gradient(data[pos_col], dt)
    
    # Calculate acceleration (derivative of velocity)
    for i, vel_col in enumerate(vel_cols):
        acc_col = acc_cols[i]
        data[acc_col + '_calc'] = np.gradient(data[vel_col + '_calc'], dt)
    
    return data

def analyze_smoothness(data, title_prefix=""):
    """Analyzes the smoothness of the data."""
    print(f"\n{title_prefix} Data Analysis:")
    
    # Analyze standard deviation of velocity (smoothness indicator)
    vel_cols = ['vx', 'vy', 'vz'] if 'vx' in data.columns else ['vx_calc', 'vy_calc', 'vz_calc']
    acc_cols = ['ax', 'ay', 'az'] if 'ax' in data.columns else ['ax_calc', 'ay_calc', 'az_calc']
    
    print("Velocity Standard Deviations:")
    for col in vel_cols:
        if col in data.columns:
            print(f"  {col}: {data[col].std():.6f}")
    
    print("Acceleration Standard Deviations:")
    for col in acc_cols:
        if col in data.columns:
            print(f"  {col}: {data[col].std():.6f}")

def plot_comparison():
    """Plots the comparison graph."""
    raw_data, smooth_data = load_data()
    
    # Calculate derivatives for raw data
    raw_data = calculate_derivatives(raw_data)
    
    # Analyze smoothness
    analyze_smoothness(raw_data, "Raw Data")
    analyze_smoothness(smooth_data, "RTS Smoothed")
    
    # Create time axis
    dt = 1/60.0
    t = np.arange(len(raw_data)) * dt
    
    # Plot comparison graph
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('RTS Smoother Comparison', fontsize=16)
    
    # Position Comparison
    pos_cols = ['x', 'y', 'z']
    for i, col in enumerate(pos_cols):
        axes[0, i].plot(t, raw_data[col], 'b-', alpha=0.7, label='Raw Data')
        axes[0, i].plot(t, smooth_data[col], 'r-', linewidth=2, label='RTS Smoothed')
        axes[0, i].set_title(f'Position {col.upper()}')
        axes[0, i].set_ylabel('Position (m)')
        axes[0, i].legend()
        axes[0, i].grid(True)
    
    # Velocity Comparison
    vel_cols = ['vx', 'vy', 'vz']
    vel_calc_cols = ['vx_calc', 'vy_calc', 'vz_calc']
    for i, (col, calc_col) in enumerate(zip(vel_cols, vel_calc_cols)):
        axes[1, i].plot(t, raw_data[calc_col], 'b-', alpha=0.7, label='Numerical Derivative')
        axes[1, i].plot(t, smooth_data[col], 'r-', linewidth=2, label='RTS Smoothed')
        axes[1, i].set_title(f'Velocity {col.upper()}')
        axes[1, i].set_ylabel('Velocity (m/s)')
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    # Acceleration Comparison
    acc_cols = ['ax', 'ay', 'az']
    acc_calc_cols = ['ax_calc', 'ay_calc', 'az_calc']
    for i, (col, calc_col) in enumerate(zip(acc_cols, acc_calc_cols)):
        axes[2, i].plot(t, raw_data[calc_col], 'b-', alpha=0.7, label='Numerical Derivative')
        axes[2, i].plot(t, smooth_data[col], 'r-', linewidth=2, label='RTS Smoothed')
        axes[2, i].set_title(f'Acceleration {col.upper()}')
        axes[2, i].set_ylabel('Acceleration (m/s²)')
        axes[2, i].set_xlabel('Time (s)')
        axes[2, i].legend()
        axes[2, i].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/rts_smoother_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved to: plots/rts_smoother_comparison.png")

if __name__ == "__main__":
    plot_comparison()