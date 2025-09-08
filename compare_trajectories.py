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
        
    def load_data(self, original_file, smoothed_file, original_2d_file=None):
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
            
            # If 2D original data is provided, load it as well
            if original_2d_file:
                self.original_2d_data = pd.read_csv(original_2d_file, header=None,
                                                 names=['x_2d', 'y_2d'])
                print(f"Successfully loaded 2D original data: {original_2d_file}")
                print(f"Data points: {len(self.original_2d_data)}")
                
            return True
            
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
    
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
            
        # Ensure data length consistency
        min_length = min(len(self.original_data), len(self.smoothed_data))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Trajectory comparison plot
        ax1 = axes[0, 0]
        ax1.plot(self.original_data['x'][:min_length],
                self.original_data['y'][:min_length],
                'b-', label='Original Trajectory', alpha=0.7, linewidth=2)
        ax1.plot(self.smoothed_data['x_smooth'][:min_length],
                self.smoothed_data['y_smooth'][:min_length],
                'r-', label='Filtered Trajectory', alpha=0.7, linewidth=2)
        
        # If 2D original data is available, plot it as well
        if self.original_2d_data is not None:
            min_length_2d = min(min_length, len(self.original_2d_data))
            ax1.plot(self.original_2d_data['x_2d'][:min_length_2d],
                    self.original_2d_data['y_2d'][:min_length_2d],
                    'g--', label='2D Original Trajectory', alpha=0.5)
        
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
        
        plt.tight_layout()
        
        # Default save path
        if not save_path:
            save_path = 'trajectory_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        # In non-interactive environment, don't call plt.show()
        print("Note: Plot has been saved as an image file, please check the generated image")
    
    def run_comparison(self, original_file, smoothed_file, original_2d_file=None,
                      save_plot=None):
        """Run complete comparison analysis"""
        print("Starting trajectory data comparison analysis...")
        
        # Load data
        if not self.load_data(original_file, smoothed_file, original_2d_file):
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
    
    # Create comparator and run analysis
    comparator = TrajectoryComparator()
    comparator.run_comparison(
        original_file=args.original,
        smoothed_file=args.smoothed,
        original_2d_file=args.original_2d,
        save_plot=args.save_plot
    )


if __name__ == "__main__":
    main()