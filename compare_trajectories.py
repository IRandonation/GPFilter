#!/usr/bin/env python3
"""
Trajectory Data Comparison Tool
Used to compare original trajectory data with filtered trajectory data
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
    """Trajectory Data Comparator"""
    
    def __init__(self):
        self.original_data = None
        self.smoothed_data = None
        self.original_2d_data = None
        self.interpolated_data = None
        self.gp_interpolation_data = None
        self.time_parameterized_data = None
        self.gp_time_parameterized_data = None
        
    def load_data(self, original_file, smoothed_file,
                  interpolated_file=None, gp_interpolation_file=None, time_parameterized_file=None,
                  gp_time_parameterized_file=None):
        """Load data files"""
        try:
            # Load original 4D trajectory data
            self.original_data = pd.read_csv(original_file, header=None,
                                           names=['x', 'y', 'z', 'param'])
            print(f"Successfully loaded original data: {original_file}")
            print(f"Data points: {len(self.original_data)}")
            
            # Load filtered trajectory data
            self.smoothed_data = pd.read_csv(smoothed_file, header=None,
                                           names=['x_smooth', 'y_smooth'])
            print(f"Successfully loaded filtered data: {smoothed_file}")
            print(f"Data points: {len(self.smoothed_data)}")
            
            # Load additional data files
            additional_files = {
                'interpolated_data': (interpolated_file, ['x_interp', 'y_interp'], None),
                'gp_interpolation_data': (gp_interpolation_file, ['x_gp', 'y_gp', 'vx_gp', 'vy_gp', 'acceleration_gp'],
                                         lambda df: df.assign(velocity_gp=np.sqrt(df['vx_gp']**2 + df['vy_gp']**2))),
                'time_parameterized_data': (time_parameterized_file, ['timestamp', 'x_time', 'y_time', 'vx_time', 'vy_time'],
                                           lambda df: (print(f"Total time: {df['timestamp'].iloc[-1]:.3f} seconds"), df)[1]),
                'gp_time_parameterized_data': (gp_time_parameterized_file, ['time', 'x', 'y', 'vx', 'vy'],
                                              self._process_gp_time_parameterized_data)
            }
            
            for attr_name, (file_path, columns, processor) in additional_files.items():
                if file_path:
                    data = pd.read_csv(file_path, header=None, names=columns)
                    print(f"Successfully loaded {attr_name}: {file_path}")
                    print(f"Data points: {len(data)}")
                    if processor:
                        data = processor(data)
                    setattr(self, attr_name, data)
            
            return True
            
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
    
    def _process_gp_time_parameterized_data(self, df):
        """Process GP time parameterized data to calculate velocity and acceleration"""
        # Calculate velocity magnitude
        df['velocity'] = np.sqrt(df['vx']**2 + df['vy']**2)
        
        # Calculate acceleration
        if len(df) >= 3:
            dt = np.diff(df['time'])
            avg_dt = np.mean(dt)
            ax = np.gradient(df['vx'], avg_dt)
            ay = np.gradient(df['vy'], avg_dt)
            df['acceleration'] = np.sqrt(ax**2 + ay**2)
        else:
            df['acceleration'] = 0.0
        
        print(f"Total time: {df['time'].iloc[-1]:.3f} seconds")
        return df
    
    def calculate_statistics(self):
        """Calculate statistics"""
        if self.original_data is None or self.smoothed_data is None:
            print("Error: Please load data first")
            return None
            
        # Ensure data length consistency
        min_length = min(len(self.original_data), len(self.smoothed_data))
        orig_x = self.original_data['x'].values[:min_length]
        orig_y = self.original_data['y'].values[:min_length]
        smooth_x = self.smoothed_data['x_smooth'].values[:min_length]
        smooth_y = self.smoothed_data['y_smooth'].values[:min_length]
        
        # Calculate errors
        error_x = orig_x - smooth_x
        error_y = orig_y - smooth_y
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
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
        """Print statistics"""
        if stats is None:
            return
            
        print("\n=== Error Statistics ===")
        print(f"X Direction Error - Mean: {stats['x_error_mean']:.6f}, Std: {stats['x_error_std']:.6f}")
        print(f"Y Direction Error - Mean: {stats['y_error_mean']:.6f}, Std: {stats['y_error_std']:.6f}")
        print(f"Magnitude Error - Mean: {stats['magnitude_error_mean']:.6f}, Std: {stats['magnitude_error_std']:.6f}")
        print(f"Max Error: {stats['max_error']:.6f}")
        print(f"Min Error: {stats['min_error']:.6f}")
    
    def plot_comparison(self, save_path=None):
        """Plot comparison"""
        if self.original_data is None or self.smoothed_data is None:
            print("Error: Please load data first")
            return
            
        # Determine if we have time parameterized data and GP interpolation data
        has_time_data = self.time_parameterized_data is not None
        has_gp_data = self.gp_interpolation_data is not None
        has_gp_time_data = self.gp_time_parameterized_data is not None
        
        # Create subplots - larger figure if we have time data or GP data
        if has_time_data or has_gp_data or has_gp_time_data:
            fig, axes = plt.subplots(4, 2, figsize=(15, 24))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Ensure data length consistency
        min_length = min(len(self.original_data), len(self.smoothed_data))
        
        # 1. Trajectory comparison plot
        ax1 = axes[0, 0]
        ax1.plot(self.original_data['x'][:min_length],
                self.original_data['y'][:min_length],
                'b-', label='Original Trajectory', alpha=0.7, linewidth=2)
        ax1.plot(self.smoothed_data['x_smooth'][:min_length],
                self.smoothed_data['y_smooth'][:min_length],
                'r-', label='Filtered Trajectory', alpha=0.7, linewidth=2)
        
        # If interpolated data is available, plot it as well
        if self.interpolated_data is not None:
            min_length_interp = min(min_length, len(self.interpolated_data))
            ax1.plot(self.interpolated_data['x_interp'][:min_length_interp],
                    self.interpolated_data['y_interp'][:min_length_interp],
                    'g--', label='Interpolated Trajectory', alpha=0.5)
        
        # If GP interpolation data is available, plot it as well
        if has_gp_data:
            ax1.plot(self.gp_interpolation_data['x_gp'],
                    self.gp_interpolation_data['y_gp'],
                    'c-.', label='GP Interpolation Trajectory', alpha=0.7)
        
        # If time parameterized data is available, plot it as well
        if has_time_data:
            ax1.plot(self.time_parameterized_data['x_time'],
                    self.time_parameterized_data['y_time'],
                    'm:', label='Time Parameterized Trajectory', alpha=0.7)
        
        # If GP time parameterized data is available, plot it as well
        if has_gp_time_data:
            ax1.plot(self.gp_time_parameterized_data['x'],
                    self.gp_time_parameterized_data['y'],
                    'r--', label='GP Time Parameterized Trajectory', alpha=0.7, linewidth=2)
        
        # If 2D original data is available, plot it as well
        if self.original_2d_data is not None:
            min_length_2d = min(min_length, len(self.original_2d_data))
            ax1.plot(self.original_2d_data['x_2d'][:min_length_2d],
                    self.original_2d_data['y_2d'][:min_length_2d],
                    'y-', label='2D Original Trajectory', alpha=0.5)
        
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. X coordinate comparison
        ax2 = axes[0, 1]
        time_steps = np.arange(min_length)
        ax2.plot(time_steps, self.original_data['x'][:min_length],
                'b-', label='Original X', alpha=0.7)
        ax2.plot(time_steps, self.smoothed_data['x_smooth'][:min_length],
                'r-', label='Filtered X', alpha=0.7)
        
        # If interpolated data is available, plot it as well
        if self.interpolated_data is not None:
            min_length_interp = min(min_length, len(self.interpolated_data))
            interp_steps = np.linspace(0, min_length-1, min_length_interp)
            ax2.plot(interp_steps, self.interpolated_data['x_interp'][:min_length_interp],
                    'g--', label='Interpolated X', alpha=0.5)
        
        # If GP interpolation data is available, plot it as well
        if has_gp_data:
            gp_steps = np.linspace(0, min_length-1, len(self.gp_interpolation_data))
            ax2.plot(gp_steps, self.gp_interpolation_data['x_gp'],
                    'c-.', label='GP Interpolation X', alpha=0.5)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('X Coordinate')
        ax2.set_title('X Coordinate Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Y coordinate comparison
        ax3 = axes[1, 0]
        ax3.plot(time_steps, self.original_data['y'][:min_length],
                'b-', label='Original Y', alpha=0.7)
        ax3.plot(time_steps, self.smoothed_data['y_smooth'][:min_length],
                'r-', label='Filtered Y', alpha=0.7)
        
        # If interpolated data is available, plot it as well
        if self.interpolated_data is not None:
            min_length_interp = min(min_length, len(self.interpolated_data))
            interp_steps = np.linspace(0, min_length-1, min_length_interp)
            ax3.plot(interp_steps, self.interpolated_data['y_interp'][:min_length_interp],
                    'g--', label='Interpolated Y', alpha=0.5)
        
        # If GP interpolation data is available, plot it as well
        if has_gp_data:
            gp_steps = np.linspace(0, min_length-1, len(self.gp_interpolation_data))
            ax3.plot(gp_steps, self.gp_interpolation_data['y_gp'],
                    'c-.', label='GP Interpolation Y', alpha=0.5)
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Y Coordinate Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Error plot
        ax4 = axes[1, 1]
        error_x = self.original_data['x'][:min_length] - self.smoothed_data['x_smooth'][:min_length]
        error_y = self.original_data['y'][:min_length] - self.smoothed_data['y_smooth'][:min_length]
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        ax4.plot(time_steps, error_x, 'r-', label='X Error', alpha=0.7)
        ax4.plot(time_steps, error_y, 'g-', label='Y Error', alpha=0.7)
        ax4.plot(time_steps, error_magnitude, 'k-', label='Error Magnitude', alpha=0.7, linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Error')
        ax4.set_title('Error Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Time parameterized plots (if available)
        if has_time_data:
            # Velocity plot
            ax5 = axes[2, 0]
            velocity = np.sqrt(self.time_parameterized_data['vx_time']**2 +
                             self.time_parameterized_data['vy_time']**2)
            ax5.plot(self.time_parameterized_data['timestamp'], velocity, 'b-', alpha=0.7)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Velocity (m/s)')
            ax5.set_title('Velocity Profile')
            ax5.grid(True, alpha=0.3)
            
            # Trajectory with time color coding
            ax6 = axes[2, 1]
            scatter = ax6.scatter(self.time_parameterized_data['x_time'],
                                self.time_parameterized_data['y_time'],
                                c=self.time_parameterized_data['timestamp'],
                                cmap='viridis', alpha=0.7)
            ax6.set_xlabel('X Coordinate')
            ax6.set_ylabel('Y Coordinate')
            ax6.set_title('Time Parameterized Trajectory')
            plt.colorbar(scatter, ax=ax6, label='Time (s)')
            ax6.grid(True, alpha=0.3)
        
        # GP time parameterized plots (if available)
        if has_gp_time_data:
            # Velocity plot
            ax7 = axes[3, 0]
            ax7.plot(self.gp_time_parameterized_data['time'],
                    self.gp_time_parameterized_data['velocity'],
                    'r-', alpha=0.7, linewidth=2, label='GP Time Parameterized Velocity')
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Velocity (m/s)')
            ax7.set_title('GP Time Parameterized Velocity Profile')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            # Acceleration plot
            ax8 = axes[3, 1]
            ax8.plot(self.gp_time_parameterized_data['time'],
                    self.gp_time_parameterized_data['acceleration'],
                    'g-', alpha=0.7, linewidth=2, label='GP Time Parameterized Acceleration')
            ax8.set_xlabel('Time (s)')
            ax8.set_ylabel('Acceleration (m/s²)')
            ax8.set_title('GP Time Parameterized Acceleration Profile')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Default save path
        if not save_path:
            save_path = 'trajectory_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        # In non-interactive environment, don't call plt.show()
        print("Note: Plot has been saved as an image file, please check the generated image")
    
    def run_comparison(self, original_file, smoothed_file, original_2d_file=None,
                      interpolated_file=None, gp_interpolation_file=None,
                      time_parameterized_file=None, gp_time_parameterized_file=None, save_plot=None):
        """Run complete comparison analysis"""
        print("Starting trajectory data comparison analysis...")
        
        # Load data
        if not self.load_data(original_file, smoothed_file,
                             interpolated_file, gp_interpolation_file,
                             time_parameterized_file, gp_time_parameterized_file):
            return False
        
        # Calculate statistics
        stats = self.calculate_statistics()
        self.print_statistics(stats)
        
        # Plot comparison
        self.plot_comparison(save_plot)
        
        print("Comparison analysis completed!")
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Trajectory Data Comparison Tool')
    parser.add_argument('--original', '-o', default='data/trajectory.csv',
                       help='Original trajectory data file path')
    parser.add_argument('--smoothed', '-s', default='output/smoothed_trajectory.csv',
                       help='Filtered trajectory data file path')
    parser.add_argument('--original-2d', '-2d', default='data/trajectory_2d.csv',
                       help='2D original trajectory data file path (optional)')
    parser.add_argument('--interpolated', '-i', default=None,
                       help='Interpolated trajectory data file path (optional)')
    parser.add_argument('--gp-interpolation', '-g', default='output/gp_interpolation_result.csv',
                       help='GP interpolation result data file path (optional)')
    parser.add_argument('--time-parameterized', '-t', default='output/time_parameterized_trajectory.csv',
                       help='Time parameterized trajectory data file path (optional)')
    parser.add_argument('--gp-time-parameterized', '-gt', default='output/gp_time_parameterized_result.csv',
                       help='GP time parameterized result data file path (optional)')
    parser.add_argument('--save-plot', '-p',
                       help='Save plot to specified path')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.original, args.smoothed]:
        if not Path(file_path).exists():
            print(f"Error: File does not exist: {file_path}")
            sys.exit(1)
    
    if args.original_2d and not Path(args.original_2d).exists():
        print(f"Warning: 2D original data file does not exist: {args.original_2d}")
        args.original_2d = None
    
    if args.interpolated and not Path(args.interpolated).exists():
        print(f"Warning: Interpolated data file does not exist: {args.interpolated}")
        args.interpolated = None
    
    if args.gp_interpolation and not Path(args.gp_interpolation).exists():
        print(f"Warning: GP interpolation data file does not exist: {args.gp_interpolation}")
        args.gp_interpolation = None
    
    if args.time_parameterized and not Path(args.time_parameterized).exists():
        print(f"Warning: Time parameterized data file does not exist: {args.time_parameterized}")
        args.time_parameterized = None
    
    if args.gp_time_parameterized and not Path(args.gp_time_parameterized).exists():
        print(f"Warning: GP time parameterized data file does not exist: {args.gp_time_parameterized}")
        args.gp_time_parameterized = None
    
    # Create comparator and run analysis
    comparator = TrajectoryComparator()
    comparator.run_comparison(
        original_file=args.original,
        smoothed_file=args.smoothed,
        original_2d_file=args.original_2d,
        interpolated_file=args.interpolated,
        gp_interpolation_file=args.gp_interpolation,
        time_parameterized_file=args.time_parameterized,
        gp_time_parameterized_file=args.gp_time_parameterized,
        save_plot=args.save_plot
    )


if __name__ == "__main__":
    main()