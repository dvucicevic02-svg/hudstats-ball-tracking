"""
predict.py — offline prediction: video file in, CSV of ball positions out.

Feeds every frame of the video, in order, through `Predictor.step()` (the
per-frame detection logic lives in predictor.py, not here). Offline means we
can afford to wait on the model at each frame, so no frame is ever skipped.
The same core also runs live via `realtime.py`, where frames may be dropped.

Each frame with a position becomes one CSV row (`frame_no,ball_x,ball_y`);
frames without the ball get no row. Rows are written as they are produced,
so stopping mid-run keeps everything up to that point. Output path is
--output, default `prediction_<size>.csv` from the config's train.size.
With --show a window plays the video with the detection drawn in (q quits).

    python -m balltrack.predict data/received/part1.mp4 \
        --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt
    python -m balltrack.predict data/received/part1.mp4 --output prediction_s.csv --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from balltrack.config import Config
from balltrack.predictor import Predictor
from balltrack.sources import frames


def predict(
    video: Path, weights: Path, cfg: Config, output: Path, show: bool = False
) -> Path:
    predictor = Predictor(weights, cfg)
    n_frames = 0
    n_rows = 0

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_no", "ball_x", "ball_y"])
        for frame_no, frame in frames(video):
            n_frames = frame_no + 1
            pos = predictor.step(frame)

            row = None
            if pos is not None:
                row = (frame_no, int(round(pos[0])), int(round(pos[1])))
                writer.writerow(row)
                n_rows += 1

            if show:
                _draw(frame, row)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show:
        cv2.destroyAllWindows()

    print(f"Wrote {n_rows} rows -> {output}  "
          f"(coverage {n_rows}/{n_frames} frames)")
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
