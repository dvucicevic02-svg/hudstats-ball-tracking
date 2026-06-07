"""
evaluate.py — honest accuracy of the predictions against ground truth.

This is the number we report. It is computed only on the held-out temporal
segment (the last `val_fraction` of the timeline), i.e. frames the model never
saw in training. Evaluating on the whole video would mix in training frames and
inflate the result.

Metrics:
  * median & mean Euclidean pixel error (median is robust to occasional misses)
  * % of frames within 10px / 20px of ground truth
  * detection rate: predicted frames / ground-truth frames, within the segment

    python -m balltrack.evaluate --pred prediction.csv --gt data/part1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from balltrack.config import Config
from balltrack.prepare_dataset import load_labels, temporal_split
from balltrack.tracking import setup_mlflow


def val_start_frame(gt: pd.DataFrame, cfg: Config) -> int:
    """First frame of the held-out segment — identical split to training."""
    kept = gt["frame_no"].values[:: cfg.data.subsample]
    _, val = temporal_split(kept, cfg.data.val_fraction, cfg.data.split_buffer)
    return int(val.min())


def evaluate(pred_path: Path, gt_path: Path, cfg: Config) -> dict:
    gt = load_labels(gt_path)
    pred = pd.read_csv(pred_path)

    start = val_start_frame(gt, cfg)
    gt_v = gt[gt["frame_no"] >= start].set_index("frame_no")
    pred_v = pred[pred["frame_no"] >= start].set_index("frame_no")

    # Pixel error only where BOTH have the frame.
    common = gt_v.index.intersection(pred_v.index)
    if len(common) == 0:
        raise ValueError("No overlapping frames in the validation segment.")
    dx = pred_v.loc[common, "ball_x"].values - gt_v.loc[common, "ball_x"].values
    dy = pred_v.loc[common, "ball_y"].values - gt_v.loc[common, "ball_y"].values
    err = np.hypot(dx, dy)  # per-frame Euclidean pixel distance: sqrt(dx^2 + dy^2)

    metrics = {
        "val_start_frame": start,
        "gt_frames": int(len(gt_v)),
        "predicted_frames": int(len(pred_v)),
        "matched_frames": int(len(common)),
        "detection_rate": round(len(common) / len(gt_v), 4),
        "median_px_error": round(float(np.median(err)), 2),
        "mean_px_error": round(float(err.mean()), 2),
        "p90_px_error": round(float(np.percentile(err, 90)), 2),
        "pct_within_10px": round(float((err <= 10).mean() * 100), 1),
        "pct_within_20px": round(float((err <= 20).mean() * 100), 1),
    }

    _print_report(metrics)
    _log_mlflow(metrics, cfg.mlflow.experiment)
    return metrics


def _print_report(m: dict) -> None:
    print("\n" + "=" * 58)
    print(" EVALUATION  (held-out temporal segment only)")
    print("=" * 58)
    print(f"  segment starts at frame : {m['val_start_frame']}")
    print(f"  ground-truth frames     : {m['gt_frames']}")
    print(f"  matched (both have it)  : {m['matched_frames']}")
    print(f"  detection rate          : {m['detection_rate']*100:.1f}%")
    print("  ---- localisation error (px) ----")
    print(f"  median                  : {m['median_px_error']}")
    print(f"  mean                    : {m['mean_px_error']}")
    print(f"  p90                     : {m['p90_px_error']}")
    print(f"  within 10px             : {m['pct_within_10px']}%")
    print(f"  within 20px             : {m['pct_within_20px']}%")
    print("=" * 58)


def _log_mlflow(m: dict, experiment: str) -> None:
    uri = setup_mlflow(experiment)
    if uri is None:
        return
    try:
        import mlflow

        with mlflow.start_run(run_name="eval"):
            mlflow.log_metrics({k: v for k, v in m.items()})
        print(f"Logged eval metrics to MLflow ({uri}).")
    except Exception as exc:
        print(f"(mlflow logging failed: {exc}; metrics above are still valid)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate predictions vs ground truth.")
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("data/part1.csv"))
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    evaluate(args.pred, args.gt, cfg)


if __name__ == "__main__":
    main()