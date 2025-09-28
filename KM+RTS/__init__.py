"""
KM+RTS: Kalman Filter + Rauch-Tung-Striebel Smoother
Online trajectory filtering and smoothing for 6DOF robot arm data
"""

from .kalman_filter_6dof import KalmanFilter6DOF
from .rts_smoother import FixedLagRTSSmoother, OnlineKMRTSProcessor

__version__ = "1.0.0"
__author__ = "Robot Arm Trajectory Processing"

__all__ = [
    'KalmanFilter6DOF',
    'FixedLagRTSSmoother', 
    'OnlineKMRTSProcessor'
]