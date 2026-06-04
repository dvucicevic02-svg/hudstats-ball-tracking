# ⚽ Soccer Ball Position Prediction

Predicts the ball's pixel coordinates in each frame of 1080p FIFA gameplay video
and writes them to `part1.csv` (`frame_no,ball_x,ball_y`).

The approach is a **lightweight YOLO26n detector fine-tuned on native-resolution
crops**, guided at inference time by an **ROI tracker + Kalman filter**. Every
design choice is grounded in the data — see `analysis/explore_dataset.py`, which
reproduces the analysis and prints the decision each finding drives.

---

## Why these choices (the short version)

| Finding (from EDA) | Decision |
|---|---|
| 27.7k labels, 96.6% coverage, 60fps → adjacent frames near-identical | **Subsample** training to every 6th frame; remove redundancy. |
| Gaps are few but huge (up to 463 frames) → scene changes, not occlusions | **Skip** them; Kalman coasts only across *short* gaps (≤8). |
| Ball speed: median 2, p99 16 px/frame; one 534px "teleport" = scene cut | Kalman tuned to ~2–3px; **outlier gate at 40px** kills HUD/player hits. |
| Ball is ~10–15px in 1080p | **Never downscale.** Train *and* infer on native-res 640px crops. |
| Adjacent frames almost duplicates | **Temporal** train/val split (not random) + buffer band → honest metric. |

The single most important idea: **the ball must never shrink.** Feeding a
downscaled full frame to the detector would crush a ~12px ball to ~3px. Both
training and inference therefore operate on full-resolution crops where the ball
keeps its true size; full-frame reacquisition runs at `imgsz=1920` (no downscale).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the data:
#   data/part1.mp4   data/part1.csv
```

## Usage

```bash
# 0. Look at the data first (prints reasoning + saves figures)
python analysis/explore_dataset.py --labels data/part1.csv --fps 60

# 1. Build the YOLO dataset (native-res crops + hard negatives, temporal split)
python -m balltrack.prepare_dataset --config configs/default.yaml

# 2. Train
python -m balltrack.train --config configs/default.yaml
# 3. Predict (final deliverable; default output = part1.csv)
python -m balltrack.predict data/part1.mp4 --output prediction.csv [--show]
# 4. Evaluate (honest metrics on the held-out segment)
python -m balltrack.evaluate --pred prediction.csv --gt data/part1.csv
```

> The final predictor writes to `--output`, **default `part1.csv`** (spec-exact).
> During development we use `--output prediction.csv` to avoid clobbering the
> ground-truth file of the same name.

---

## Project layout

```
analysis/explore_dataset.py   EDA + reasoning (run me first)
configs/default.yaml          all tunables, overridable
src/balltrack/
  config.py                   typed config (dataclasses)
  prepare_dataset.py          video+labels -> YOLO dataset  [done]
  train.py                    YOLO26n fine-tune             [done]
  tracker.py                  ROI + Kalman post-processing  [done]
  predict.py                  video -> part1.csv (+ --show) [done]
  evaluate.py                 pixel-error metrics + MLflow  [done]
  viz.py                      detection overlay             [next]
```

## Evaluation metric

Reported **only on the held-out temporal split**: median & mean Euclidean pixel
error, % of frames within 10px / 20px, and detection rate (coverage). The true
test is the held-out `part2` the reviewers run `predict.py` on.