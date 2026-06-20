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
ROI crop in predictor.py.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class BallTracker:
    def __init__(self, max_jump: float = 25.0, max_coast: int = 8):
        self.max_jump = max_jump
        self.max_coast = max_coast

        # State = [x, y, vx, vy]; we measure [x, y]. One step = one frame,
        # so velocity is in px/frame and elapsed time is counted in steps.
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        # R: doubt in the yolo detector — small, its centre is good to ~2px.
        kf.R = np.eye(2) * 3.0
        # Q: how fast truth can change — position barely, velocity more
        # sized to the EDA's ~2-3 px/frame real motion.
        kf.Q = np.diag([1.0, 1.0, 4.0, 4.0])
        # P: initial uncertainty — huge, so first measurements dominate.
        kf.P = np.eye(4) * 500.0
        self.kf = kf

        self.initialized = False
        self.coast = 0  # consecutive frames without an accepted measurement

    @property
    def position(self) -> tuple[float, float]:
        x = np.ravel(self.kf.x) 
        return float(x[0]), float(x[1])

    def predict(self, n_steps: int = 1) -> tuple[float, float]:
        """Advance the state `n_steps` frames; returns the predicted position.

        `n_steps` > 1 means the source dropped frames since the last call
        (live mode); the filter must propagate through them or its prediction
        lags behind the real ball.
        """
        if not self.initialized:
            return self.position
        for _ in range(n_steps):
            self.kf.predict()
        return self.position

    def is_outlier(self, meas: tuple[float, float], n_steps: int = 1) -> bool:
        """True if `meas` is too far from where the ball should be.
        """
        if not self.initialized:
            return False  # nothing to compare against yet
        px, py = self.position
        return float(np.hypot(meas[0] - px, meas[1] - py)) > self.max_jump * n_steps

    def update(self, meas: tuple[float, float]) -> None:
        """Accept a measurement and correct the state."""
        if not self.initialized:
            self.kf.x = np.array([meas[0], meas[1], 0.0, 0.0])
            self.initialized = True
        else:
            self.kf.update(np.array(meas, dtype=float))
        self.coast = 0

    def mark_missing(self, n_steps: int = 1) -> bool:
        """Call when no measurement was accepted this frame.
        """
        self.coast += n_steps
        return self.coast <= self.max_coast

    @property
    def alive(self) -> bool:
        return self.initialized and self.coast <= self.max_coast

    def reset(self) -> None:
        self.initialized = False
        self.coast = 0
        self.kf.P = np.eye(4) * 500.0