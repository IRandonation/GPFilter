import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import os

# Set matplotlib to use English fonts to avoid character rendering issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class TrajectoryVisualizer:
    """
    Visualizer for comparing original, filtered, and corrected trajectory data
    """
    
    def __init__(self, original_file: str, data1_file: str, data2_file: str, dt: float = 0.001):
        """
        Initialize visualizer with data files
        
        Args:
            original_file: Path to original trajectory CSV
            data1_file: Path to filtered data (data1)
            data2_file: Path to corrected data (data2)
            dt: Time step for calculations
        """
        self.dt = dt
        self.load_data(original_file, data1_file, data2_file)
        self.calculate_derivatives()
    
    def load_data(self, original_file: str, data1_file: str, data2_file: str) -> None:
        """Load all data files"""
        # Load original data
        self.original = pd.read_csv(original_file, header=None)
        self.original.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        
        # Load filtered data (data1)
        self.data1 = pd.read_csv(data1_file)
        
        # Load corrected data (data2)
        self.data2 = pd.read_csv(data2_file)
        
        # Create time vector
        self.time = np.arange(len(self.original)) * self.dt
        
        print(f"Loaded {len(self.original)} data points")
    
    def calculate_derivatives(self) -> None:
        """Calculate derivatives for original and data2"""
        coords = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        
        # Calculate original velocities and accelerations using finite differences
        self.original_vel = pd.DataFrame()
        self.original_acc = pd.DataFrame()
        
        for coord in coords:
            # Velocity (first derivative)
            vel = np.gradient(self.original[coord], self.dt)
            self.original_vel[f'v{coord}'] = vel
            
            # Acceleration (second derivative)
            acc = np.gradient(vel, self.dt)
            self.original_acc[f'a{coord}'] = acc
        
        # Calculate data2 derivatives
        self.data2_vel = pd.DataFrame()
        self.data2_acc = pd.DataFrame()
        
        for coord in coords:
            # Velocity (first derivative)
            vel = np.gradient(self.data2[coord], self.dt)
            self.data2_vel[f'v{coord}'] = vel
            
            # Acceleration (second derivative)
            acc = np.gradient(vel, self.dt)
            self.data2_acc[f'a{coord}'] = acc
        
        # Calculate data1 position from velocity integration
        self.data1_pos_integrated = pd.DataFrame()
        for coord in coords:
            # Integrate velocity to get position
            integrated_pos = np.cumsum(self.data1[f'v{coord}']) * self.dt
            # Add initial position
            integrated_pos += self.original[coord].iloc[0]
            self.data1_pos_integrated[coord] = integrated_pos
        
        # Calculate data1 velocity from position differentiation
        self.data1_vel_diff = pd.DataFrame()
        for coord in coords:
            vel = np.gradient(self.data1[coord], self.dt)
            self.data1_vel_diff[f'v{coord}'] = vel
        
        # Calculate data1 acceleration from velocity differentiation
        self.data1_acc_diff = pd.DataFrame()
        for coord in coords:
            acc = np.gradient(self.data1[f'v{coord}'], self.dt)
            self.data1_acc_diff[f'a{coord}'] = acc
    
    def plot_position_comparison(self, save_dir: str = ".") -> None:
        """Plot position comparisons for all coordinates"""
        coords = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        coord_names = ['X Position', 'Y Position', 'Z Position', 
                      'RX Rotation', 'RY Rotation', 'RZ Rotation']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (coord, name) in enumerate(zip(coords, coord_names)):
            ax = axes[i]
            
            # Plot comparisons
            ax.plot(self.time, self.original[coord], 'b-', label='Original Position', alpha=0.7)
            ax.plot(self.time, self.data1[coord], 'r-', label='Data1 Position (Filtered)', alpha=0.8)
            ax.plot(self.time, self.data2[coord], 'g-', label='Data2 Position (Corrected)', alpha=0.8)
            ax.plot(self.time, self.data1_pos_integrated[coord], 'm--', 
                   label='Data1 Velocity Integrated', alpha=0.6)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{name} Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'position_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_velocity_comparison(self, save_dir: str = ".") -> None:
        """Plot velocity comparisons for all coordinates"""
        coords = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        coord_names = ['X Velocity', 'Y Velocity', 'Z Velocity', 
                      'RX Angular Velocity', 'RY Angular Velocity', 'RZ Angular Velocity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (coord, name) in enumerate(zip(coords, coord_names)):
            ax = axes[i]
            
            # Plot comparisons
            ax.plot(self.time, self.original_vel[f'v{coord}'], 'b-', 
                   label='Original Differential Velocity', alpha=0.7)
            ax.plot(self.time, self.data1[f'v{coord}'], 'r-', 
                   label='Data1 Velocity (Filtered)', alpha=0.8)
            ax.plot(self.time, self.data1_vel_diff[f'v{coord}'], 'orange', 
                   label='Data1 Position Differential', alpha=0.6)
            ax.plot(self.time, self.data2_vel[f'v{coord}'], 'g--', 
                   label='Data2 Differential Velocity', alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{name} Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'velocity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_acceleration_comparison(self, save_dir: str = ".") -> None:
        """Plot acceleration comparisons for all coordinates"""
        coords = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        coord_names = ['X Acceleration', 'Y Acceleration', 'Z Acceleration', 
                      'RX Angular Acceleration', 'RY Angular Acceleration', 'RZ Angular Acceleration']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (coord, name) in enumerate(zip(coords, coord_names)):
            ax = axes[i]
            
            # Plot comparisons
            ax.plot(self.time, self.original_acc[f'a{coord}'], 'b-', 
                   label='Original Differential Acceleration', alpha=0.7)
            ax.plot(self.time, self.data1[f'a{coord}'], 'r-', 
                   label='Data1 Acceleration (Filtered)', alpha=0.8)
            ax.plot(self.time, self.data1_acc_diff[f'a{coord}'], 'orange', 
                   label='Data1 Velocity Differential', alpha=0.6)
            ax.plot(self.time, self.data2_acc[f'a{coord}'], 'g--', 
                   label='Data2 Differential Acceleration', alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{name} Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'acceleration_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, save_dir: str = ".") -> None:
        """Plot error analysis between different methods"""
        coords = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, coord in enumerate(coords):
            ax = axes[i]
            
            # Calculate errors
            error_data1_vs_original = self.data1[coord] - self.original[coord]
            error_data2_vs_original = self.data2[coord] - self.original[coord]
            error_integrated_vs_original = self.data1_pos_integrated[coord] - self.original[coord]
            
            ax.plot(self.time, error_data1_vs_original, 'r-', 
                   label='Data1 vs Original', alpha=0.8)
            ax.plot(self.time, error_data2_vs_original, 'g-', 
                   label='Data2 vs Original', alpha=0.8)
            ax.plot(self.time, error_integrated_vs_original, 'm--', 
                   label='Integrated vs Original', alpha=0.6)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{coord.upper()} Error')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{coord.upper()} Error Analysis')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self, save_dir: str = ".") -> None:
        """Generate all comparison plots"""
        print("Generating position comparison plots...")
        self.plot_position_comparison(save_dir)
        
        print("Generating velocity comparison plots...")
        self.plot_velocity_comparison(save_dir)
        
        print("Generating acceleration comparison plots...")
        self.plot_acceleration_comparison(save_dir)
        
        print("Generating error analysis plots...")
        self.plot_error_analysis(save_dir)
        
        print(f"All plots saved to {save_dir}")


def main():
    """Main function to run visualization"""
    # File paths
    original_file = "../data/trajectory.csv"
    output_data_dir = "../output"
    data1_file = os.path.join(output_data_dir, "data1_filtered.csv")
    data2_file = os.path.join(output_data_dir, "data2_corrected.csv")
    output_plots_dir = "../plots"
    
    # Check if processed data exists
    if not os.path.exists(data1_file) or not os.path.exists(data2_file):
        print("Processed data files not found. Please run kalman_filter.py or main.py first.")
        return
    
    # Create visualizer
    visualizer = TrajectoryVisualizer(original_file, data1_file, data2_file)
    
    # Generate all plots
    visualizer.generate_all_plots(output_plots_dir)


if __name__ == "__main__":
    main()