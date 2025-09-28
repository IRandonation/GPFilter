import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrajectoryKalmanFilter:
    """
    Kalman Filter for trajectory data with position, velocity, and acceleration estimation.
    
    State vector: [x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz, ax, ay, az, arx, ary, arz]
    - Position: x, y, z, rx, ry, rz (6 DOF)
    - Velocity: vx, vy, vz, vrx, vry, vrz (6 DOF)
    - Acceleration: ax, ay, az, arx, ary, arz (6 DOF)
    """
    
    def __init__(self, dt: float = 1/60, process_noise_pos: float = 0.01, process_noise_vel: float = 0.01, process_noise_acc: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize Kalman Filter
        
        Args:
            dt: Time step (default 1ms for 1kHz)
            process_noise_pos: Process noise variance for position
            process_noise_vel: Process noise variance for velocity
            process_noise_acc: Process noise variance for acceleration
            measurement_noise: Measurement noise variance
        """
        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.process_noise_acc = process_noise_acc
        self.measurement_noise = measurement_noise
    
        # State vector: [x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz, ax, ay, az, arx, ary, arz]
        self.state_dim = 18
        # Measurement vector: [x, y, z, rx, ry, rz]
        self.measurement_dim = 6
    
        self.F = self._create_state_transition_matrix()
        self.H = self._create_measurement_matrix()
        self.Q = self._create_process_noise_matrix()
        self.R = self._create_measurement_noise_matrix()
    
        self.x = np.zeros((self.state_dim, 1))  # Initial state estimate
        self.P = np.eye(self.state_dim) * 1.0  # Initial covariance matrix
        self.I = np.eye(self.state_dim) # Identity matrix for Kalman gain calculation
        
    def _create_measurement_matrix(self) -> np.ndarray:
        """Create measurement matrix (we only observe position)"""
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[:6, :6] = np.eye(6)  # Only position is observed
        return H
    
    def _create_measurement_noise_matrix(self) -> np.ndarray:
        """Create measurement noise covariance matrix"""
        return np.eye(self.measurement_dim) * self.measurement_noise
    
    def _create_state_transition_matrix(self) -> np.ndarray:
        """Create state transition matrix for constant acceleration model"""
        F = np.eye(self.state_dim)
        dt = self.dt
        dt2 = dt * dt / 2
        
        # For each DOF (6 total), set up the kinematic relationships
        for i in range(6):
            pos_idx = i
            vel_idx = i + 6
            acc_idx = i + 12
            
            # Position = position + velocity*dt + 0.5*acceleration*dt^2
            F[pos_idx, vel_idx] = dt
            F[pos_idx, acc_idx] = dt2
            
            # Velocity = velocity + acceleration*dt
            F[vel_idx, acc_idx] = dt
            
        return F
    
    def _create_process_noise_matrix(self):
        Q = np.eye(self.state_dim)
    
        # Position noise (x, y, z, rx, ry, rz)
        Q[0:6, 0:6] *= self.process_noise_pos
    
        # Velocity noise (vx, vy, vz, vrx, vry, vrz)
        Q[6:12, 6:12] *= self.process_noise_vel
    
        # Acceleration noise (ax, ay, az, arx, ary, arz)
        Q[12:18, 12:18] *= self.process_noise_acc # Acceleration noise is often higher
        return Q
    
    def predict(self) -> None:
        """Prediction step of Kalman filter"""
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray) -> None:
        """Update step of Kalman filter"""
        # Innovation
        y = measurement.reshape(-1, 1) - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (self.I - K @ self.H) @ self.P
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current position, velocity, and acceleration estimates"""
        position = self.x[:6].flatten()
        velocity = self.x[6:12].flatten()
        acceleration = self.x[12:18].flatten()
        
        return position, velocity, acceleration
    
    def set_initial_state(self, initial_position: np.ndarray) -> None:
        """Set initial state from first measurement"""
        self.x[:6] = initial_position.reshape(-1, 1)
        # Initialize velocity and acceleration to zero
        self.x[6:] = 0


class ProgressiveCorrection:
    """
    Progressive correction algorithm to smooth position errors over N frames
    """
    
    def __init__(self, correction_frames: int = 50, correction_type: str = 'linear'):
        """
        Initialize progressive correction
        
        Args:
            correction_frames: Number of frames to distribute correction over
            correction_type: 'linear' or 'exponential' decay
        """
        self.correction_frames = correction_frames
        self.correction_type = correction_type
        self.pending_corrections = []
    
    def add_correction(self, error: np.ndarray, frame_idx: int) -> None:
        """Add a new correction to be applied progressively"""
        if self.correction_type == 'linear':
            weights = np.linspace(1.0, 0.0, self.correction_frames)
        else:  # exponential
            weights = np.exp(-np.linspace(0, 3, self.correction_frames))
            weights = weights / weights.sum()
        
        correction_per_frame = error.reshape(-1, 1) * weights.reshape(1, -1) / self.correction_frames
        
        for i, correction in enumerate(correction_per_frame.T):
            self.pending_corrections.append({
                'frame': frame_idx + i,
                'correction': correction
            })
    
    def get_correction(self, frame_idx: int) -> np.ndarray:
        """Get total correction for current frame"""
        total_correction = np.zeros(6)
        
        # Apply all pending corrections for this frame
        remaining_corrections = []
        for correction_data in self.pending_corrections:
            if correction_data['frame'] == frame_idx:
                total_correction += correction_data['correction']
            elif correction_data['frame'] > frame_idx:
                remaining_corrections.append(correction_data)
        
        self.pending_corrections = remaining_corrections
        return total_correction


def process_trajectory_data(input_file: str, output_dir: str, dt: float = 0.001, process_noise_pos=0.01, process_noise_vel=0.01, process_noise_acc=0.01, measurement_noise=0.1,
                            correction_frames=50, correction_type='linear', error_threshold=0.001):
    """
    Process trajectory data with Kalman filter and progressive correction
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save output files
        dt: Time step
    
    Returns:
        Tuple of (data1_file_path, data2_file_path)
    """
    logger.info(f"Loading trajectory data from {input_file}")
    
    # Load data
    data = pd.read_csv(input_file, header=None)
    data.columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    
    n_points = len(data)
    logger.info(f"Processing {n_points} data points")
    
    # Initialize Kalman filter
    kf = TrajectoryKalmanFilter(dt, process_noise_pos, process_noise_vel, process_noise_acc, measurement_noise)
    
    # Initialize progressive correction
    pc = ProgressiveCorrection(correction_frames=50)
    
    # Storage for results
    data1_results = {
        'x': [], 'y': [], 'z': [], 'rx': [], 'ry': [], 'rz': [],
        'vx': [], 'vy': [], 'vz': [], 'vrx': [], 'vry': [], 'vrz': [],
        'ax': [], 'ay': [], 'az': [], 'arx': [], 'ary': [], 'arz': []
    }
    
    data2_results = {
        'x': [], 'y': [], 'z': [], 'rx': [], 'ry': [], 'rz': []
    }
    
    # Set initial state
    initial_measurement = data.iloc[0].values
    kf.set_initial_state(initial_measurement)
    
    logger.info("Starting Kalman filtering...")
    
    for i in range(n_points):
        # Predict
        kf.predict()
        
        # Update with measurement
        measurement = data.iloc[i].values
        kf.update(measurement)
        
        # Get filtered state
        position, velocity, acceleration = kf.get_state()
        
        # Store data1 (filtered position, velocity, acceleration)
        for j, coord in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz']):
            data1_results[coord].append(position[j])
            data1_results[f'v{coord}'].append(velocity[j])
            data1_results[f'a{coord}'].append(acceleration[j])
        
        # Calculate position error for progressive correction
        if i > 0:
            # Integrate velocity to get position
            prev_position = np.array([data1_results['x'][-2], data1_results['y'][-2], 
                                    data1_results['z'][-2], data1_results['rx'][-2], 
                                    data1_results['ry'][-2], data1_results['rz'][-2]])
            integrated_position = prev_position + velocity * dt
            position_error = position - integrated_position
            
            # Add correction if error is significant
            if np.linalg.norm(position_error[:3]) > 0.001:  # 1mm threshold
                pc.add_correction(position_error, i)
        
        # Apply progressive correction for data2
        correction = pc.get_correction(i)
        corrected_position = position + correction
        
        # Store data2 (progressively corrected position only)
        for j, coord in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz']):
            data2_results[coord].append(corrected_position[j])
        
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{n_points} points")
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data1 (position, velocity, acceleration)
    data1_df = pd.DataFrame(data1_results)
    data1_file = os.path.join(output_dir, 'data1_filtered.csv')
    data1_df.to_csv(data1_file, index=False)
    logger.info(f"Saved data1 to {data1_file}")
    
    # Save data2 (corrected position only)
    data2_df = pd.DataFrame(data2_results)
    data2_file = os.path.join(output_dir, 'data2_corrected.csv')
    data2_df.to_csv(data2_file, index=False)
    logger.info(f"Saved data2 to {data2_file}")
    
    return data1_file, data2_file


if __name__ == "__main__":
    input_file = "../data/trajectory.csv"
    output_dir = "../output"
    
    data1_file, data2_file = process_trajectory_data(input_file, output_dir)
    print(f"Processing complete!")
    print(f"Data1 (filtered): {data1_file}")
    print(f"Data2 (corrected): {data2_file}")