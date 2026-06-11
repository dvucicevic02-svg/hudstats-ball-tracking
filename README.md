# ⚽ Soccer Ball Position Prediction

Predicts the ball's pixel coordinates in each frame of 1080p FIFA gameplay video
and writes them to `part1.csv` (`frame_no,ball_x,ball_y`).

The approach is a **YOLO26 detector fine-tuned on native-resolution crops**,
guided at inference time by an **ROI tracker and Kalman filter**. The model size
is configurable (`n`, `s`, `m`, `l`, `x` via `train.size`), so the
accuracy/speed trade-off can be tuned without touching code, and each size keeps
its own weights, predictions, and MLflow runs for side-by-side comparison. Every
design choice is grounded in the data. See `analysis/explore_dataset.py`, which
reproduces the analysis and prints the decision each finding drives.

---

## From data to decisions

Before writing any model code, we inspected the dataset. Each decision below
follows from a measured property of the data, not a generic recipe.

| Finding (from EDA) | Decision |
|---|---|
| 27.7k labels, 96.6% coverage, 60fps, so adjacent frames are near-identical | Subsample training to every 6th frame to remove redundancy. |
| Gaps are few but large (up to 463 frames), so they are scene changes, not occlusions | Skip large gaps; let the Kalman filter coast only across short ones (up to 8 frames). |
| Ball speed is 2 px/frame median, 16 at p99, with one 534px "teleport" that is a scene cut | Tune the Kalman filter to roughly 2 to 3 px/frame, and set an outlier gate at 40px to reject false hits on the HUD, radar, or players. |
| The ball is about 11px in 1080p (measured with `analysis/measure_ball.py`) | Set the training box to about 22px, and never downscale the ball. |
| Adjacent frames are almost duplicates | Split train and validation by time, not randomly, with a buffer band between them, so the metric is honest. |
| The ball is small, fast, and often on bright lines or in crowds | Train on native-resolution 640px crops, add jitter so the ball is not always centred, and mine hard-negative crops so the model learns what is not a ball. |

## The core idea: the ball must never shrink

The most important decision is that the ball keeps its true size everywhere.
Feeding a downscaled full frame to the detector would crush an 11px ball to about
3px and destroy the signal. So both training and inference operate on
full-resolution 640px crops, where the ball stays sharp. The two stages share one
crop size (a single source of truth in `crop.py`), so their scales can never
diverge. When the track is lost, full-frame reacquisition runs at `imgsz=1920`,
which is also free of downscaling.

## How it works, end to end

The pipeline is a sequence of independent stages, each consuming the previous
one's output:

1. **Inspect the data** (`explore_dataset.py`) and measure the ball (`measure_ball.py`).
2. **Prepare the dataset** (`prepare_dataset.py`): cut native-resolution crops, convert each centre point to a normalised YOLO box, subsample, jitter, add hard negatives, and split temporally.
3. **Verify the crops** (`check_crops.py`): draw the YOLO boxes back onto the saved crops, so you can confirm by eye that the box lands on the ball, and that hard negatives contain no ball, before spending a training run on them.
4. **Train** (`train.py`): fine-tune a COCO-pretrained YOLO26 model, adapting it to a single ball class. Hyper-parameters, metrics, and `best.pt` are logged to MLflow through `tracking.py`.
5. **Predict**: run the model over the video. The per-frame core (`predictor.py`) uses the ROI tracker and Kalman filter (`tracker.py`) to predict where the ball should be, crops an ROI there and detects at native scale; impossible jumps are rejected and short gaps are bridged. Two drivers share this one core: **offline** (`predict.py`) processes every frame in order, and **real-time** (`realtime.py`) replays the video at its native FPS like a live camera — if the model cannot keep up, stale frames are dropped (latency stays bounded) and the tracker is told how many frames really passed (`n_steps`), so its per-frame limits stay honest. Both stream rows to the CSV as they are produced.
6. **Evaluate** (`evaluate.py`): compare against ground truth, but only on the held-out temporal segment the model never saw, which is the only honest measure. The metrics are logged to MLflow through `tracking.py`.

Steps 2 and 5 both cut their crops through the same `crop.py` (one shared crop
size, a single source of truth), so the ball is seen at the exact same native
scale during training and inference and the two can never drift apart.

## Results

On the held-out segment, the **YOLO26s** model scores a median pixel error of
2.83 and a mean of 4.43, with 92.4% of frames within 10px and 97.7% within 20px,
at a 99.0% detection rate. The gap between median and mean reflects a tail of
harder frames (motion-blurred ball, ball on a white line, or in a crowd).

---

## Quick start
> _Place the video and labels under `data/received/` first (`part1.mp4`, `part1.csv`)._


```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the package and its dependencies (declared in pyproject.toml)
pip install -e .

# 3. Start the MLflow tracking server (in its own terminal)
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

# 4. Predict with the s model — two variants of the same per-frame core:

# 4a. OFFLINE: every frame, in order -> the full CSV (add --show to watch)
python -m balltrack.predict data/received/part1.mp4 --show \
    --output prediction_s.csv \
    --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt

# 4b. REAL-TIME: replay at native FPS like a live camera; frames the model
#     can't keep up with are dropped (drop % in the window, q to quit)
python -m balltrack.realtime data/received/part1.mp4 \
    --output live.csv \
    --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt
```

The MLflow UI at <http://127.0.0.1:5000> shows the `n`, `s`, and `m` training
runs and their evaluations together in the `balltrack` experiment. Select several

and click **Compare** to put their metrics side by side. It is also worth opening
a training run's **Artifacts** tab to see the detailed plots of how training
progressed (loss and metric curves, PR and F1 curves, confusion matrix) along
with the other logged metrics, and the matching evaluation runs.


---

## Detailed usage

Place the video and labels under `data/received/` first (`part1.mp4`,
`part1.csv`), then run the full pipeline. The commands below use the `s` model as
the example; switch variants by changing `train.size` in `configs/default.yaml`.

```bash
# 1. Look at the data first (prints analysis conclusions + saves figures)
python analysis/explore_dataset.py --labels data/received/part1.csv --fps 60

# 2. Measure the ball size to justify box_size
python analysis/measure_ball.py --video data/received/part1.mp4 --labels data/received/part1.csv

# 3. Build the YOLO dataset (native-res crops + hard negatives, temporal split)
python -m balltrack.prepare_dataset --config configs/default.yaml

# 4. Visually verify the boxes/labels land on the ball before training
python analysis/check_crops.py --dataset data/yolo --split train --n 16

# 5. Train the variant set by train.size (logs to MLflow as yolo26<size>_ball)
python -m balltrack.train --config configs/default.yaml

# 6. Predict -> prediction_<size>.csv
python -m balltrack.predict data/received/part1.mp4 \
    --output prediction_s.csv \
    --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt

# 7. Evaluate (honest metrics on the held-out segment only, logs as eval_<size>)
python -m balltrack.evaluate --pred prediction_s.csv --gt data/received/part1.csv --config configs/default.yaml

# 8. Real-time variant: replay the video at native FPS like a live camera
#    (green box tracks the ball, drop % in the corner, press q to quit)
python -m balltrack.realtime data/received/part1.mp4 \
    --output live.csv \
    --weights runs/detect/outputs/train/yolo26s_ball/weights/best.pt
```

---

## Project layout

```
analysis/explore_dataset.py   EDA + reasoning 
configs/default.yaml          all tunables, overridable
src/balltrack/
  config.py                   typed config (dataclasses)
  crop.py                     shared crop geometry (train + infer)
  prepare_dataset.py          video+labels -> YOLO dataset
  train.py                    YOLO26 fine-tune + MLflow
  tracker.py                  Kalman filter: smooth, gate, coast
  predictor.py                per-frame core (ROI + reacquire), shared by both modes
  sources.py                  frame sources: file iterator, paced latest-frame reader
  predict.py                  offline: video -> CSV (+ --show)
  realtime.py                 live replay: native-FPS window + CSV, bounded latency
  evaluate.py                 pixel-error metrics + MLflow
  tracking.py                 MLflow server setup
```