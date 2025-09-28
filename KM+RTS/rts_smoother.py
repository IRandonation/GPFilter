import numpy as np
from typing import List, Dict, Optional

from kalman_filter_6dof import KalmanFilter6DOF


class FixedLagRTSSmoother:
    """
    Fixed-Lag Rauch-Tung-Striebel (RTS) Smoother for online trajectory smoothing.
    """

    def __init__(self, kalman_filter: KalmanFilter6DOF, lag: int):
        self.kf = kalman_filter
        self.lag = lag

        self.x_history = []  # Stores x_post from KF
        self.P_history = []  # Stores P_post from KF
        self.F_history = []  # Stores F from KF
        self.Q_history = []  # Stores Q from KF

        self.smoothed_states = []
        self.smoothed_covariances = []

    def _rts_backward_pass(self, k: int):
        # Ensure k is a valid index for the history
        if k < 0 or k >= len(self.x_history) - 1:
            return

        x_k_post = self.x_history[k]
        P_k_post = self.P_history[k]
        F_k = self.F_history[k]
        Q_k = self.Q_history[k]

        # Predict next state from current smoothed state
        x_kplus1_pred = F_k @ x_k_post
        P_kplus1_pred = F_k @ P_k_post @ F_k.T + Q_k

        # If P_kplus1_pred is singular or ill-conditioned, add a small diagonal term
        if np.linalg.cond(P_kplus1_pred) > 1 / np.finfo(P_kplus1_pred.dtype).eps:
            P_kplus1_pred += np.eye(P_kplus1_pred.shape[0]) * 1e-9

        # Smoother gain
        C_k = P_k_post @ F_k.T @ np.linalg.inv(P_kplus1_pred)

        # Smoothed state and covariance
        x_k_smooth = x_k_post + C_k @ (self.smoothed_states[-1] - x_kplus1_pred)
        P_k_smooth = P_k_post + C_k @ (self.smoothed_covariances[-1] - P_kplus1_pred) @ C_k.T

        self.smoothed_states[-1] = x_k_smooth
        self.smoothed_covariances[-1] = P_k_smooth

    def update(self, x_post: np.ndarray, P_post: np.ndarray, F: np.ndarray, Q: np.ndarray):
        self.x_history.append(x_post)
        self.P_history.append(P_post)
        self.F_history.append(F)
        self.Q_history.append(Q)

        # If we have enough history, perform smoothing
        if len(self.x_history) > self.lag:
            # Initialize the latest smoothed state with the KF posterior estimate
            if len(self.smoothed_states) == 0:
                self.smoothed_states.append(self.x_history[-1])
                self.smoothed_covariances.append(self.P_history[-1])
            else:
                # The latest smoothed state is always the current KF posterior before backward pass
                self.smoothed_states.append(self.x_history[-1])
                self.smoothed_covariances.append(self.P_history[-1])

            # Perform backward pass for the relevant window
            for k in range(len(self.x_history) - 2, len(self.x_history) - self.lag - 1, -1):
                self._rts_backward_pass(k)

            # Remove oldest history if it's beyond the lag window and already smoothed
            if len(self.x_history) > self.lag + 1:
                self.x_history.pop(0)
                self.P_history.pop(0)
                self.F_history.pop(0)
                self.Q_history.pop(0)
                self.smoothed_states.pop(0)
                self.smoothed_covariances.pop(0)


class OnlineKMRTSProcessor:
    """
    Combines Kalman Filter and Fixed-Lag RTS Smoother for online processing.
    """

    def __init__(
        self,
        lag: int,
        dt: float,
        pos_process_noise: float = 1.0,
        vel_process_noise: float = 1.0,
        acc_process_noise: float = 1.0,
        pos_measurement_noise: float = 1.0,
        vel_measurement_noise: float = 1.0,
        acc_measurement_noise: float = 1.0,
        initial_position: Optional[np.ndarray] = None
    ):
        self.kf = KalmanFilter6DOF(
            dt=dt,
            pos_process_noise=pos_process_noise,
            vel_process_noise=vel_process_noise,
            acc_process_noise=acc_process_noise,
            pos_measurement_noise=pos_measurement_noise,
            vel_measurement_noise=vel_measurement_noise,
            acc_measurement_noise=acc_measurement_noise,
            initial_position=initial_position
        )
        self.rts_smoother = FixedLagRTSSmoother(self.kf, lag)
        self.lag = lag

        self.filtered_positions = []
        self.filtered_velocities = []
        self.filtered_accelerations = []

        self.smoothed_positions = []
        self.smoothed_velocities = []
        self.smoothed_accelerations = []

        self.timestamps = []

    def process_measurement(self, measurement: np.ndarray, timestamp: float):
        # Kalman Filter Prediction
        self.kf.predict()

        # Kalman Filter Update
        self.kf.update(measurement)

        # Store KF history for RTS
        self.kf.store_history(measurement)

        # RTS Smoother Update
        self.rts_smoother.update(
            self.kf.x.copy(),
            self.kf.P.copy(),
            self.kf.F.copy(),
            self.kf.Q.copy()
        )

        # Store filtered results
        self.filtered_positions.append(self.kf.get_position())
        self.filtered_velocities.append(self.kf.get_velocity())
        self.filtered_accelerations.append(self.kf.get_acceleration())
        self.timestamps.append(timestamp)

        # Store smoothed results if available
        if len(self.rts_smoother.smoothed_states) > 0:
            smoothed_state = self.rts_smoother.smoothed_states[0]
            self.smoothed_positions.append(smoothed_state[:6].flatten())
            self.smoothed_velocities.append(smoothed_state[6:12].flatten())
            self.smoothed_accelerations.append(smoothed_state[12:18].flatten())

        return {
            "filtered_position": self.kf.get_position(),
            "filtered_velocity": self.kf.get_velocity(),
            "filtered_acceleration": self.kf.get_acceleration(),
            "smoothed_position": self.smoothed_positions[-1] if self.smoothed_positions else None,
            "smoothed_velocity": self.smoothed_velocities[-1] if self.smoothed_velocities else None,
            "smoothed_acceleration": self.smoothed_accelerations[-1] if self.smoothed_accelerations else None,
        }

    def get_results_summary(self) -> Dict:
        return {
            "filtered_positions": np.array(self.filtered_positions),
            "filtered_velocities": np.array(self.filtered_velocities),
            "filtered_accelerations": np.array(self.filtered_accelerations),
            "smoothed_positions": np.array(self.smoothed_positions),
            "smoothed_velocities": np.array(self.smoothed_velocities),
            "smoothed_accelerations": np.array(self.smoothed_accelerations),
            "timestamps": np.array(self.timestamps),
        }