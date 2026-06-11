"""
predictor.py — the causal per-frame inference core.

`Predictor.step(frame)` is one tick of the pipeline: ROI-crop where the Kalman
filter expects the ball (native resolution, the ball never shrinks), or scan
the full frame at imgsz=1920 to re-acquire a lost track, then run the result
through the outlier gate / coast logic in `tracker.py`.

It is deliberately free of any I/O (no video capture, no CSV): offline
prediction (`predict.py`) and live prediction (`realtime.py`) both drive this
one class, so the detection logic cannot diverge between the two, the same
single-source-of-truth idea as `crop.py`.

Live sources may drop frames when the model can't keep up, pass the number of
source frames elapsed since the previous call as `n_steps` so the Kalman
prediction, the outlier gate (px/frame) and the coast budget stay consistent
with real time. Offline, every frame is processed and `n_steps` stays 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from balltrack.config import Config
from balltrack.crop import extract_crop
from balltrack.tracker import BallTracker


def _best_detection(result, conf: float) -> tuple[float, float, float] | None:
    """Return (cx, cy, confidence) of the most confident box, or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    keep = confs >= conf
    if not keep.any():
        return None
    xywh = boxes.xywh.cpu().numpy()[keep]
    confs = confs[keep]
    i = int(confs.argmax())
    return float(xywh[i, 0]), float(xywh[i, 1]), float(confs[i])


class Predictor:
    """Position pipeline: Yolo detector + ROI + Kalman tracker"""

    def __init__(self, weights: Path, cfg: Config):
        from ultralytics import YOLO 
        
        self.model = YOLO(str(weights))
        self.track_cfg = cfg.track
        self.crop_size = cfg.data.crop_size  # SHARED with training (single source of truth)
        self.tracker = BallTracker(max_jump=cfg.track.max_jump,
                                   max_coast=cfg.track.max_coast)

    def step(self, frame: np.ndarray,
             n_steps: int = 1) -> tuple[float, float] | None:
        """Process one frame; return the ball position or None (no ball).

        `n_steps` = source frames elapsed since the previous call (1 when no
        frames were dropped). It scales the Kalman prediction, the outlier
        gate and the coast budget so dropped frames don't kill the track.
        """
        t = self.track_cfg

        if self.tracker.alive:
            # ROI path: look where the ball is predicted to be 
            px, py = self.tracker.predict(n_steps)
            crop, x0, y0 = extract_crop(frame, px, py, self.crop_size)
            det = _best_detection(self.model(crop, imgsz=self.crop_size,
                                             conf=t.conf, verbose=False)[0], t.conf)
            # measurement: map crop-local detection back to full-frame coords
            meas = (x0 + det[0], y0 + det[1]) if det else None
        else:
            # reacquisition: full frame, native scale (1920x1080)
            det = _best_detection(self.model(frame, imgsz=t.reacquire_imgsz,
                                             conf=t.conf, verbose=False)[0], t.conf)
            meas = (det[0], det[1]) if det else None

        # post-processing decision 
        if meas is not None and not self.tracker.is_outlier(meas, n_steps):
            self.tracker.update(meas)
            return self.tracker.position

        # no usable detection: coast across a short gap, else let it die (skip)
        if self.tracker.alive and self.tracker.initialized:
            if self.tracker.mark_missing(n_steps):
                return self.tracker.position
            self.tracker.reset()
        # not yet initialised: we simply have no ball yet -> skip frame
        return None
