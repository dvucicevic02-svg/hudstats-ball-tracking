"""
predict.py — run the full pipeline on a video and write ball coordinates.

For each frame:
  * If the track is alive, crop a native-resolution ROI around the Kalman
    prediction and detect there (fast, and the ball keeps its true size).
  * Otherwise (start, or after a long gap) scan the full frame at imgsz=1920
    (still no downscaling) to re-acquire the ball.
  * Run the detection through the tracker's outlier gate, accept & smooth, or
    coast across a short gap, or if the gap is long, skip (scene change).

Output: a CSV `frame_no,ball_x,ball_y` (one row per frame with a position),
written to --output, default `prediction_<size>.csv` (derived from the config's
train.size). Use --show to watch the detections frame-by-frame, in the style of
show_ball_dataset.py (press q to quit).

    python -m balltrack.predict data/received/part1.mp4 \
        --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt
    python -m balltrack.predict data/received/part1.mp4 --output prediction_s.csv --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from balltrack.config import Config
from balltrack.crop import extract_crop
from balltrack.tracker import BallTracker


def _best_detection(result, conf: float):
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

def predict(
    video: Path, weights: Path, cfg: Config, output: Path, show: bool = False
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    t = cfg.track
    crop_size = cfg.data.crop_size  # SHARED with training (single source of truth)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    tracker = BallTracker(max_jump=t.max_jump, max_coast=t.max_coast)
    rows: list[tuple[int, int, int]] = []
    frame_no = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if tracker.alive:
            # --- ROI path: look where the ball is predicted to be ---
            px, py = tracker.predict()
            crop, x0, y0 = extract_crop(frame, px, py, crop_size)
            det = _best_detection(model(crop, imgsz=crop_size, conf=t.conf,
                                        verbose=False)[0], t.conf)
            meas = (x0 + det[0], y0 + det[1]) if det else None # measurement: map crop-local detection back to full-frame coords
        else:
            # --- reacquisition: full frame, native scale (1920x1080) ---
            det = _best_detection(model(frame, imgsz=t.reacquire_imgsz,
                                        conf=t.conf, verbose=False)[0], t.conf)
            meas = (det[0], det[1]) if det else None

        # --- post-processing decision ---
        if meas is not None and not tracker.is_outlier(meas):
            tracker.update(meas)
            bx, by = tracker.position
            rows.append((frame_no, int(round(bx)), int(round(by))))
        else:
            # no usable detection: coast across a short gap, else let it die (skip)
            if tracker.alive and tracker.initialized:
                if tracker.mark_missing():
                    bx, by = tracker.position
                    rows.append((frame_no, int(round(bx)), int(round(by))))
                else:
                    tracker.reset()
            # if not yet initialised, we simply have no ball yet -> skip frame

        if show:
            _draw(frame, rows[-1] if rows and rows[-1][0] == frame_no else None)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows, columns=["frame_no", "ball_x", "ball_y"])
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows -> {output}  "
          f"(coverage {len(df)}/{frame_no + 1} frames)")
    return output


def _draw(frame: np.ndarray, row: tuple[int, int, int] | None) -> None:
    """Overlay the detection (for --show) in the style of show_ball_dataset.py"""
    if row is not None:
        _, bx, by = row
        cv2.rectangle(frame, (bx - 5, by - 5), (bx + 5, by + 5), (0, 255, 0), 2)
    show_img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Prediction (q to quit)", show_img)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict ball positions in a video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    weights = args.weights or (
        Path("outputs/train") / cfg.train.run_name / "weights" / "best.pt")
    output = args.output or Path(f"prediction_{cfg.train.size}.csv")
    predict(args.video, weights, cfg, output, show=args.show)


if __name__ == "__main__":
    main()