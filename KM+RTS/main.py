"""
Main program for KM+RTS online filtering
Processes trajectory data with different lag values and generates comparison results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import List, Dict, Optional
import time
from typing import Optional, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalman_filter_6dof import KalmanFilter6DOF
from rts_smoother import OnlineKMRTSProcessor


def load_trajectory_data(file_path: str) -> pd.DataFrame:
    """Load trajectory data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded trajectory data: {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_output_directories():
    """Create output directories if they don't exist"""
    dirs = ['output', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")


def compute_numerical_derivatives(positions: np.ndarray, dt: float) -> tuple:
    """
    Compute numerical derivatives for velocity and acceleration
    
    Args:
        positions: Position array (N x 6)
        dt: Time step
        
    Returns:
        velocities, accelerations
    """
    n_samples = len(positions)
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    
    # Forward difference for first point
    if n_samples > 1:
        velocities[0] = (positions[1] - positions[0]) / dt
    
    # Central difference for middle points
    for i in range(1, n_samples - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    
    # Backward difference for last point
    if n_samples > 1:
        velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    # Compute accelerations from velocities
    if n_samples > 1:
        accelerations[0] = (velocities[1] - velocities[0]) / dt
    
    for i in range(1, n_samples - 1):
        accelerations[i] = (velocities[i + 1] - velocities[i - 1]) / (2 * dt)
    
    if n_samples > 1:
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    
    return velocities, accelerations


def calculate_rms_error(original: np.ndarray, filtered: np.ndarray) -> float:
    """Calculate RMS error between original and filtered data"""
    if len(original) != len(filtered):
        min_len = min(len(original), len(filtered))
        original = original[:min_len]
        filtered = filtered[:min_len]
    
    return np.sqrt(np.mean((original - filtered) ** 2))


def process_with_lag(
    data: pd.DataFrame,
    lag: int,
    dt: float,
    initial_position: Optional[np.ndarray] = None,
    pos_process_noise: float = 1.0,
    vel_process_noise: float = 1.0,
    acc_process_noise: float = 1.0,
    pos_measurement_noise: float = 1.0,
    vel_measurement_noise: float = 1.0,
    acc_measurement_noise: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    print(f"\nProcessing with lag = {lag}")

    processor = OnlineKMRTSProcessor(
        lag=lag,
        dt=dt,
        pos_process_noise=pos_process_noise,
        vel_process_noise=vel_process_noise,
        acc_process_noise=acc_process_noise,
        pos_measurement_noise=pos_measurement_noise,
        vel_measurement_noise=vel_measurement_noise,
        acc_measurement_noise=acc_measurement_noise,
        initial_position=initial_position
    )

    filtered_data = []
    smoothed_data = []
    
    # Extract position columns (assuming x, y, z, rx, ry, rz)
    position_cols = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    if not all(col in data.columns for col in position_cols):
        # Try alternative column names
        if 'X' in data.columns:
            position_cols = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        else:
            # Use first 6 columns
            position_cols = data.columns[:6].tolist()
    
    positions = data[position_cols].values
    n_samples = len(positions)
    
    # Process each measurement online
    results = []
    for i in range(n_samples):
        measurement = positions[i]
        result = processor.process_measurement(measurement, i * dt)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{n_samples} samples")
    
    # Get final results
    summary = processor.get_results_summary()
    
    # Compute numerical derivatives for original data
    orig_velocities, orig_accelerations = compute_numerical_derivatives(positions, dt)
    
    return {
        'lag': lag,
        'original_positions': positions,
        'original_velocities': orig_velocities,
        'original_accelerations': orig_accelerations,
        'filtered_positions': summary['filtered_positions'],
        'filtered_velocities': summary['filtered_velocities'],
        'filtered_accelerations': summary['filtered_accelerations'],
        'smoothed_positions': summary['smoothed_positions'],
        'smoothed_velocities': summary['smoothed_velocities'],
        'smoothed_accelerations': summary['smoothed_accelerations'],
        'timestamps': summary['timestamps']
    }


def save_results_to_csv(results: Dict, lag: int):
    """Save processing results to CSV files"""
    lag_str = f"lag_{lag}"
    
    # Original data
    orig_df = pd.DataFrame(results['original_positions'], 
                          columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    orig_df.to_csv(f'output/original_positions_{lag_str}.csv', index=False)
    
    # Filtered data
    filt_df = pd.DataFrame(results['filtered_positions'], 
                          columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    filt_df.to_csv(f'output/filtered_positions_{lag_str}.csv', index=False)
    
    # Smoothed data (if available)
    if len(results['smoothed_positions']) > 0:
        smooth_df = pd.DataFrame(results['smoothed_positions'], 
                                columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
        smooth_df.to_csv(f'output/smoothed_positions_{lag_str}.csv', index=False)
    
    # Velocities
    orig_vel_df = pd.DataFrame(results['original_velocities'], 
                              columns=['vx', 'vy', 'vz', 'vrx', 'vry', 'vrz'])
    orig_vel_df.to_csv(f'output/original_velocities_{lag_str}.csv', index=False)
    
    filt_vel_df = pd.DataFrame(results['filtered_velocities'], 
                              columns=['vx', 'vy', 'vz', 'vrx', 'vry', 'vrz'])
    filt_vel_df.to_csv(f'output/filtered_velocities_{lag_str}.csv', index=False)
    
    if len(results['smoothed_velocities']) > 0:
        smooth_vel_df = pd.DataFrame(results['smoothed_velocities'], 
                                    columns=['vx', 'vy', 'vz', 'vrx', 'vry', 'vrz'])
        smooth_vel_df.to_csv(f'output/smoothed_velocities_{lag_str}.csv', index=False)
    
    # Accelerations
    orig_acc_df = pd.DataFrame(results['original_accelerations'], 
                              columns=['ax', 'ay', 'az', 'arx', 'ary', 'arz'])
    orig_acc_df.to_csv(f'output/original_accelerations_{lag_str}.csv', index=False)
    
    filt_acc_df = pd.DataFrame(results['filtered_accelerations'], 
                              columns=['ax', 'ay', 'az', 'arx', 'ary', 'arz'])
    filt_acc_df.to_csv(f'output/filtered_accelerations_{lag_str}.csv', index=False)
    
    if len(results['smoothed_accelerations']) > 0:
        smooth_acc_df = pd.DataFrame(results['smoothed_accelerations'], 
                                    columns=['ax', 'ay', 'az', 'arx', 'ary', 'arz'])
        smooth_acc_df.to_csv(f'output/smoothed_accelerations_{lag_str}.csv', index=False)
    
    print(f"Saved results for lag {lag} to CSV files")


def create_comparison_plots(results: Dict, lag: int):
    """Create comparison plots for position, velocity, and acceleration"""
    lag_str = f"lag_{lag}"
    timestamps = results['timestamps']
    
    # Position comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Position Comparison (Lag = {lag})', fontsize=16)
    
    pos_labels = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        ax.plot(timestamps, results['original_positions'][:, i], 'b-', 
                label='Original', alpha=0.7, linewidth=1)
        ax.plot(timestamps, results['filtered_positions'][:, i], 'r-', 
                label='Filtered', alpha=0.8, linewidth=1.5)
        
        if len(results['smoothed_positions']) > 0:
            smooth_timestamps = timestamps[lag:]  # Account for delay
            ax.plot(smooth_timestamps, results['smoothed_positions'][:, i], 'g-', 
                    label='Smoothed', alpha=0.9, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(pos_labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/position_comparison_{lag_str}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Velocity comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Velocity Comparison (Lag = {lag})', fontsize=16)
    
    vel_labels = ['VX', 'VY', 'VZ', 'VRX', 'VRY', 'VRZ']
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        ax.plot(timestamps, results['original_velocities'][:, i], 'b-', 
                label='Original', alpha=0.7, linewidth=1)
        ax.plot(timestamps, results['filtered_velocities'][:, i], 'r-', 
                label='Filtered', alpha=0.8, linewidth=1.5)
        
        if len(results['smoothed_velocities']) > 0:
            smooth_timestamps = timestamps[lag:]
            ax.plot(smooth_timestamps, results['smoothed_velocities'][:, i], 'g-', 
                    label='Smoothed', alpha=0.9, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(vel_labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/velocity_comparison_{lag_str}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Acceleration comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Acceleration Comparison (Lag = {lag})', fontsize=16)
    
    acc_labels = ['AX', 'AY', 'AZ', 'ARX', 'ARY', 'ARZ']
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        ax.plot(timestamps, results['original_accelerations'][:, i], 'b-', 
                label='Original', alpha=0.7, linewidth=1)
        ax.plot(timestamps, results['filtered_accelerations'][:, i], 'r-', 
                label='Filtered', alpha=0.8, linewidth=1.5)
        
        if len(results['smoothed_accelerations']) > 0:
            smooth_timestamps = timestamps[lag:]
            ax.plot(smooth_timestamps, results['smoothed_accelerations'][:, i], 'g-', 
                    label='Smoothed', alpha=0.9, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(acc_labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/acceleration_comparison_{lag_str}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created comparison plots for lag {lag}")


def calculate_performance_metrics(results: Dict) -> Dict:
    """Calculate performance metrics"""
    metrics = {}
    
    # Position RMS errors
    pos_rms_filtered = calculate_rms_error(
        results['original_positions'], 
        results['filtered_positions']
    )
    metrics['position_rms_filtered'] = pos_rms_filtered
    
    if len(results['smoothed_positions']) > 0:
        # Align arrays for comparison (account for lag)
        lag = results['lag']
        orig_aligned = results['original_positions'][lag:]
        pos_rms_smoothed = calculate_rms_error(orig_aligned, results['smoothed_positions'])
        metrics['position_rms_smoothed'] = pos_rms_smoothed
    
    # Velocity RMS errors
    vel_rms_filtered = calculate_rms_error(
        results['original_velocities'], 
        results['filtered_velocities']
    )
    metrics['velocity_rms_filtered'] = vel_rms_filtered
    
    if len(results['smoothed_velocities']) > 0:
        lag = results['lag']
        vel_aligned = results['original_velocities'][lag:]
        vel_rms_smoothed = calculate_rms_error(vel_aligned, results['smoothed_velocities'])
        metrics['velocity_rms_smoothed'] = vel_rms_smoothed
    
    return metrics


def main():
    """Main function"""
    print("KM+RTS Online Filter - Starting processing...")
    
    # Create output directories
    create_output_directories()
    
    # Load trajectory data
    data_path = '../data/trajectory.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    data = load_trajectory_data(data_path)
    if data is None:
        return

    # Extract position columns (assuming x, y, z, rx, ry, rz)
    position_cols = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    if not all(col in data.columns for col in position_cols):
        # Try alternative column names
        if 'X' in data.columns:
            position_cols = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        else:
            # Use first 6 columns
            position_cols = data.columns[:6].tolist()
    
    measurements = data[position_cols].values
    initial_position = measurements[0]

    # Test different lag values
    lag_values = [3, 5, 8]
    dt = 1/60.0  # 60Hz

    # Default noise parameters
    pos_process_noise = 1e-5
    vel_process_noise = 1e-3
    acc_process_noise = 1e-2
    pos_measurement_noise = 1e-4
    vel_measurement_noise = 1e-4
    acc_measurement_noise = 1e-4

    all_results = {}
    all_metrics = {}

    for lag in lag_values:
        print(f"\n{'='*50}")
        print(f"Processing with lag = {lag}")
        print(f"{'='*50}")
        
        # Process data
        results = process_with_lag(
            data,
            lag,
            dt,
            initial_position,
            pos_process_noise,
            vel_process_noise,
            acc_process_noise,
            pos_measurement_noise,
            vel_measurement_noise,
            acc_measurement_noise
        )
        all_results[lag] = results
        
        # Save results
        save_results_to_csv(results, lag)
        
        # Create plots
        create_comparison_plots(results, lag)
        
        # Calculate metrics
        metrics = calculate_performance_metrics(results)
        all_metrics[lag] = metrics
        
        print(f"\nPerformance Metrics for lag {lag}:")
        print(f"Position RMS (Filtered): {metrics['position_rms_filtered']:.6f}")
        if 'position_rms_smoothed' in metrics:
            print(f"Position RMS (Smoothed): {metrics['position_rms_smoothed']:.6f}")
        print(f"Velocity RMS (Filtered): {metrics['velocity_rms_filtered']:.6f}")
        if 'velocity_rms_smoothed' in metrics:
            print(f"Velocity RMS (Smoothed): {metrics['velocity_rms_smoothed']:.6f}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Lag':<5} {'Pos RMS (Filt)':<15} {'Pos RMS (Smooth)':<16} {'Vel RMS (Filt)':<15} {'Vel RMS (Smooth)':<16}")
    print("-" * 80)
    
    for lag in lag_values:
        metrics = all_metrics[lag]
        pos_filt = metrics['position_rms_filtered']
        pos_smooth = metrics.get('position_rms_smoothed', 0)
        vel_filt = metrics['velocity_rms_filtered']
        vel_smooth = metrics.get('velocity_rms_smoothed', 0)
        
        print(f"{lag:<5} {pos_filt:<15.6f} {pos_smooth:<16.6f} {vel_filt:<15.6f} {vel_smooth:<16.6f}")
    
    print("\nProcessing completed successfully!")
    print(f"Results saved to 'output/' directory")
    print(f"Plots saved to 'plots/' directory")


if __name__ == "__main__":
    main()