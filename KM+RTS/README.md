# KM+RTS: Kalman Filter + Rauch-Tung-Striebel Smoother

## Overview

This package implements an online trajectory filtering and smoothing system for 6DOF robot arm data using:

- **Kalman Filter (KM)**: Real-time state estimation with position, velocity, and acceleration states
- **Fixed-Lag RTS Smoother**: Rauch-Tung-Striebel smoothing with configurable delay for improved trajectory quality

## Features

- **6DOF Support**: Handles position and orientation (x, y, z, rx, ry, rz)
- **Online Processing**: Real-time filtering at 60Hz with configurable lag
- **Multiple Lag Testing**: Compares different lag values (3, 5, 8 samples)
- **Comprehensive Output**: Generates CSV files and comparison plots
- **Performance Metrics**: Calculates RMS errors for evaluation

## File Structure

```
KM+RTS/
├── __init__.py                 # Package initialization
├── kalman_filter_6dof.py      # 6DOF Kalman Filter implementation
├── rts_smoother.py            # Fixed-lag RTS Smoother
├── main.py                    # Main processing script
├── output/                    # Generated CSV files
│   ├── original_*.csv         # Original trajectory data
│   ├── filtered_*.csv         # Kalman filtered results
│   └── smoothed_*.csv         # RTS smoothed results
└── plots/                     # Generated comparison plots
    ├── position_comparison_*.png
    ├── velocity_comparison_*.png
    └── acceleration_comparison_*.png
```

## Usage

### Running the Main Program

```bash
cd KM+RTS
uv run main.py
```

### Using the Classes Directly

```python
from kalman_filter_6dof import KalmanFilter6DOF
from rts_smoother import OnlineKMRTSProcessor

# Initialize processor with lag=5
processor = OnlineKMRTSProcessor(lag=5, dt=1/60.0)

# Process measurements online
for measurement in trajectory_data:
    result = processor.process_measurement(measurement)
    
    # Access filtered and smoothed results
    filtered_pos = result['filtered']['position']
    if result['smoothed'] is not None:
        smoothed_pos = result['smoothed']['position']
```

## Algorithm Details

### Kalman Filter

- **State Vector**: [position(6), velocity(6), acceleration(6)] = 18 dimensions
- **Motion Model**: Constant acceleration model
- **Measurements**: Position-only observations (6DOF)
- **Update Rate**: 60Hz (dt = 1/60 seconds)

### RTS Smoother

- **Fixed-Lag**: Configurable delay (L samples)
- **Window-Based**: Maintains sliding window of states
- **Backward Pass**: Applies RTS equations for optimal smoothing
- **Online Output**: Provides smoothed results with L-sample delay

## Performance Results

Based on the test run with 299 trajectory samples:

| Lag | Position RMS (Filtered) | Position RMS (Smoothed) | Velocity RMS (Filtered) | Velocity RMS (Smoothed) |
|-----|------------------------|------------------------|------------------------|------------------------|
| 3   | 0.644955              | 0.570471              | 9.284311              | 9.255044              |
| 5   | 0.644955              | 0.562611              | 9.284311              | 9.265137              |
| 8   | 0.644955              | 0.574827              | 9.284311              | 9.282567              |

**Key Observations:**
- RTS smoothing consistently improves position accuracy
- Lag=5 provides the best position smoothing performance
- Velocity improvements are more modest but still beneficial
- Longer lags don't always provide better results due to trajectory dynamics

## Configuration Parameters

### Kalman Filter Parameters
- `dt`: Time step (default: 1/60.0 for 60Hz)
- `process_noise`: Process noise variance (default: 1e-4)
- `measurement_noise`: Measurement noise variance (default: 1e-2)

### RTS Smoother Parameters
- `lag`: Fixed lag in samples (tested: 3, 5, 8)

## Input Data Format

The system expects CSV data with 6DOF position columns:
- Position: x, y, z (or X, Y, Z)
- Orientation: rx, ry, rz (or RX, RY, RZ)

## Output Files

### CSV Files (per lag value)
- `original_positions_lag_X.csv`: Input trajectory data
- `filtered_positions_lag_X.csv`: Kalman filtered positions
- `smoothed_positions_lag_X.csv`: RTS smoothed positions
- Similar files for velocities and accelerations

### Plots (per lag value)
- `position_comparison_lag_X.png`: Position trajectories comparison
- `velocity_comparison_lag_X.png`: Velocity profiles comparison
- `acceleration_comparison_lag_X.png`: Acceleration profiles comparison

## Dependencies

- numpy: Numerical computations
- pandas: Data handling
- matplotlib: Plotting
- scipy: Scientific computing (if needed)

## Installation

Ensure you have the required dependencies:

```bash
uv add numpy pandas matplotlib
```

## Notes

- The system automatically handles different column naming conventions
- Numerical derivatives are computed using central differences
- RTS smoothing introduces a fixed delay equal to the lag parameter
- Performance metrics account for the alignment offset due to lag