#!/usr/bin/env python3
"""
Main script for Kalman Filter trajectory processing with progressive correction
"""

import os
import sys
import logging
from src.kalman_filter import process_trajectory_data
from src.visualize_results import TrajectoryVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    logger.info("Starting Kalman Filter trajectory processing...")
    
    # File paths
    input_file = "../data/trajectory.csv"
    output_data_dir = "./output"
    output_plots_dir = "./plots"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        # Step 1: Process trajectory data with Kalman filter
        logger.info("Step 1: Processing trajectory data with Kalman filter...")
        data1_file, data2_file = process_trajectory_data(
            input_file=input_file,
            output_dir=output_data_dir,
            dt=1/60,
            process_noise_pos=1e-5,  # Default value for position process noise
            process_noise_vel=1e-4,  # Default value for velocity process noise
            process_noise_acc=1e-3,  # Default value for acceleration process noise
            measurement_noise=1e-4
        )
        
        logger.info("Kalman filtering completed successfully!")
        logger.info(f"Data1 (filtered): {data1_file}")
        logger.info(f"Data2 (corrected): {data2_file}")
        
        # Step 2: Generate visualizations
        logger.info("Step 2: Generating visualization plots...")
        
        visualizer = TrajectoryVisualizer(
            original_file=input_file,
            data1_file=data1_file,
            data2_file=data2_file,
            dt=1/60
        )
        
        # Generate all comparison plots
        visualizer.generate_all_plots(output_plots_dir)
        
        logger.info("Visualization completed successfully!")
        
        # Summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"Input file: {input_file}")
        print(f"Output data directory: {os.path.abspath(output_data_dir)}")
        print(f"Output plots directory: {os.path.abspath(output_plots_dir)}")
        print("\nGenerated files:")
        print(f"  - {os.path.basename(data1_file)} (Filtered data with position, velocity, acceleration)")
        print(f"  - {os.path.basename(data2_file)} (Progressively corrected position data)")
        print("\nGenerated plots:")
        print("  - position_comparison.png")
        print("  - velocity_comparison.png") 
        print("  - acceleration_comparison.png")
        print("  - error_analysis.png")
        print("\nAll files saved to:", os.path.abspath(output_data_dir), "and", os.path.abspath(output_plots_dir))
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()