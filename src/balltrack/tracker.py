"""
tracker.py — temporal post-processing for the per-frame detector.

A constant-velocity Kalman filter turns noisy, occasionally-wrong per-frame
detections into a smooth, robust track. It does three jobs, all justified by the
EDA (real motion ~2-3 px/frame, p99 ~16 px/frame, gaps are scene cuts):

  1. SMOOTH    — average out detector jitter.
  2. GATE      — reject detections that imply an impossible jump (> max_jump),
                 e.g. a spurious hit on the radar / HUD / a player.
  3. COAST     — when a frame has no usable detection, ride the prediction
                 forward for up to `max_coast` frames (short occlusion), then
                 give up (large gap = scene change -> we simply skip).

The filter also tells the inference loop WHERE to look next, which drives the
ROI crop in predict.py.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class BallTracker:
    def __init__(self, max_jump: float = 40.0, max_coast: int = 8, dt: float = 1.0):
        self.max_jump = max_jump
        self.max_coast = max_coast

        # State = [x, y, vx, vy]; we measure [x, y].
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        # Measurement noise: detector centre is good to a couple of px.
        kf.R = np.eye(2) * 3.0
        # Process noise: scaled to ~2-3 px/frame real motion (EDA).
        kf.Q = np.diag([1.0, 1.0, 4.0, 4.0])
        # large initial uncertainty
        kf.P = np.eye(4) * 500.0 
        self.kf = kf

        self.initialized = False
        self.coast = 0  # consecutive frames without an accepted measurement

    @property
    def position(self) -> tuple[float, float]:
        x = np.ravel(self.kf.x) 
        return float(x[0]), float(x[1])

    def predict(self) -> tuple[float, float]:
        """Advance the state one frame; returns the predicted position (prior)."""
        if not self.initialized:
            return self.position
        self.kf.predict()
        return self.position

    def is_outlier(self, meas: tuple[float, float]) -> bool:
        """True if `meas` is too far from where the ball should be."""
        if not self.initialized:
            return False  # nothing to compare against yet
        px, py = self.position
        return float(np.hypot(meas[0] - px, meas[1] - py)) > self.max_jump

    def update(self, meas: tuple[float, float]) -> None:
        """Accept a measurement and correct the state."""
        if not self.initialized:
            self.kf.x = np.array([meas[0], meas[1], 0.0, 0.0])
            self.initialized = True
        else:
            self.kf.update(np.array(meas, dtype=float))
        self.coast = 0

    def mark_missing(self) -> bool:
        """Call when no measurement was accepted this frame.

        Returns True while the track is still alive (coasting), False once the
        gap is too long to bridge, at which point the caller resets/skips.
        """
        self.coast += 1
        return self.coast <= self.max_coast

    @property
    def alive(self) -> bool:
        return self.initialized and self.coast <= self.max_coast

    def reset(self) -> None:
        self.initialized = False
        self.coast = 0
        self.kf.P = np.eye(4) * 500.0