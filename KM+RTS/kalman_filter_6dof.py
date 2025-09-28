import numpy as np
from typing import Optional

class KalmanFilter6DOF:
    """
    6DOF Kalman Filter for position, velocity, and acceleration.
    State vector: [x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz, ax, ay, az, arx, ary, arz]
    Measurement vector: [x, y, z, rx, ry, rz]
    """

    def __init__(
        self,
        dt: float,
        pos_process_noise: float = 1.0,
        vel_process_noise: float = 1.0,
        acc_process_noise: float = 1.0,
        pos_measurement_noise: float = 1.0,
        vel_measurement_noise: float = 1.0,
        acc_measurement_noise: float = 1.0,
        initial_position: Optional[np.ndarray] = None
    ):
        self.dt = dt
        self.state_dim = 18  # 6 (pos) + 6 (vel) + 6 (acc)
        self.measurement_dim = 6  # 6 (pos)

        # State vector: [x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz, ax, ay, az, arx, ary, arz]
        self.x = np.zeros((self.state_dim, 1))
        if initial_position is not None:
            self.x[:6, 0] = initial_position

        self.P = np.eye(self.state_dim) * 1.0  # State covariance matrix

        self.F = self._create_state_transition_matrix()  # State transition matrix
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Measurement matrix
        self.H[:6, :6] = np.eye(6)

        self.Q = self._create_process_noise_covariance(pos_process_noise, vel_process_noise, acc_process_noise)  # Process noise covariance
        self.R = self._create_measurement_noise_covariance(pos_measurement_noise, vel_measurement_noise, acc_measurement_noise)  # Measurement noise covariance

        # History for RTS smoother
        self.state_history = []
        self.covariance_history = []
        self.measurement_history = []

    def _create_state_transition_matrix(self) -> np.ndarray:
        F = np.eye(self.state_dim)
        # Position from velocity
        F[:6, 6:12] = np.eye(6) * self.dt
        # Position from acceleration
        F[:6, 12:18] = np.eye(6) * 0.5 * self.dt**2
        # Velocity from acceleration
        F[6:12, 12:18] = np.eye(6) * self.dt
        return F

    def _create_process_noise_covariance(self, pos_noise: float, vel_noise: float, acc_noise: float) -> np.ndarray:
        Q = np.eye(self.state_dim) * 1e-9  # Small default noise
        
        # Position noise (integrated from acceleration noise)
        Q[:6, :6] = (self.dt**4 / 4) * acc_noise * np.eye(6)
        # Cross terms position-velocity
        Q[:6, 6:12] = (self.dt**3 / 2) * acc_noise * np.eye(6)
        Q[6:12, :6] = (self.dt**3 / 2) * acc_noise * np.eye(6)
        # Velocity noise (integrated from acceleration noise)
        Q[6:12, 6:12] = (self.dt**2) * acc_noise * np.eye(6)
        # Acceleration noise
        Q[12:18, 12:18] = acc_noise * np.eye(6)
        
        # Apply individual noise scaling
        Q[:6, :6] *= pos_noise
        Q[6:12, 6:12] *= vel_noise
        Q[12:18, 12:18] *= acc_noise

        return Q

    def _create_measurement_noise_covariance(self, pos_noise: float, vel_noise: float, acc_noise: float) -> np.ndarray:
        R = np.eye(self.measurement_dim)
        R[:6, :6] = pos_noise * np.eye(6)  # Only position is measured
        return R

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        Y = z.reshape(-1, 1) - (self.H @ self.x)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ Y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.flatten()

    def get_position(self) -> np.ndarray:
        return self.x[:6].flatten()

    def get_velocity(self) -> np.ndarray:
        return self.x[6:12].flatten()

    def get_acceleration(self) -> np.ndarray:
        return self.x[12:18].flatten()

    def store_history(self, z: np.ndarray):
        self.state_history.append(self.x)
        self.covariance_history.append(self.P)
        self.measurement_history.append(z)

    def get_history(self):
        return (
            np.array([s.flatten() for s in self.state_history]),
            np.array(self.covariance_history),
            np.array(self.measurement_history)
        )