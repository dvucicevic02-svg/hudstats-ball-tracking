"""
realtime.py — live prediction: the video plays at real speed, the model keeps up.

Runs the same `Predictor.step()` as predict.py, but live the model cannot ask
the video to wait: frames arrive at native FPS (`LatestFrameReader`) and only
the newest one is served. If a frame takes too long, the frames that arrived
meanwhile are dropped, the prediction stays about now, never about a backlog.
The number of dropped frames goes to the tracker as `n_steps`, so the Kalman
filter and its outlier/coast limits account for the time that really passed.

A window plays the feed with the detection drawn in, plus the drop rate
0% means the model keeps up (q quits). The window is the product here; the
CSV is an optional extra: pass --output to also record the coordinates, one
row (`frame_no,ball_x,ball_y`) per processed frame with a position, dropped
frames simply have no row. Rows are written as they are produced, so
stopping mid-run keeps everything so far.

    python -m balltrack.realtime data/received/part1.mp4 \
        --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt \
        --output live.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from balltrack.config import Config
from balltrack.predictor import Predictor
from balltrack.sources import LatestFrameReader


def run(video: Path, weights: Path, cfg: Config,
        output: Path | None = None) -> None:
    predictor = Predictor(weights, cfg)

    # CSV is optional here (the window is the product); buffering=1 -> flush
    # per row, so a run stopped with q or Ctrl+C keeps everything so far.
    csv_file = output.open("w", newline="", buffering=1) if output else None
    writer = csv.writer(csv_file) if csv_file else None
    if writer:
        writer.writerow(["frame_no", "ball_x", "ball_y"])

    last_no = -1
    n_processed = 0

    try:
        with LatestFrameReader(video) as reader:
            while True:
                item = reader.read()
                if item is None:  # source ended
                    break
                frame_no, frame = item
                n_steps = frame_no - last_no  # >1 means frames were dropped
                last_no = frame_no
                n_processed += 1

                pos = predictor.step(frame, n_steps=n_steps)
                if pos is not None and writer:
                    writer.writerow(
                        (frame_no, int(round(pos[0])), int(round(pos[1]))))

                drop_rate = 1 - n_processed / (frame_no + 1)
                _draw(frame, pos, drop_rate)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass 
    finally:
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()

    total = last_no + 1
    print(f"Processed {n_processed}/{total} source frames "
          f"({total - n_processed} dropped)"
          + (f" -> {output}" if output else ""))


def _draw(frame: np.ndarray, pos: tuple[float, float] | None,
          drop_rate: float) -> None:
    """Detection overlay plus the drop rate (0% = the model keeps up)."""
    if pos is not None:
        bx, by = int(round(pos[0])), int(round(pos[1]))
        cv2.rectangle(frame, (bx - 5, by - 5), (bx + 5, by + 5), (0, 255, 0), 2)
    cv2.putText(frame, f"drop {drop_rate:4.0%}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    show_img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Realtime (q to quit)", show_img)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay a recording as a live feed and predict in real time.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None,
                    help="optional CSV; omit to run display-only")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    weights = args.weights or (
        Path("outputs/train") / cfg.train.run_name / "weights" / "best.pt")
    run(args.video, weights, cfg, output=args.output)


if __name__ == "__main__":
    main()
