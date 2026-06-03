# CLAUDE.md — balltrack

Soccer ball position prediction in 1080p FIFA gameplay video. Input: a video.
Output: `part1.csv` (`frame_no,ball_x,ball_y`). This is a job-assignment repo;
it is graded on **accuracy, simplicity, engineering quality, maintainability**.

## Commands
```bash
pip install -e .                                          # install package + deps
python -m balltrack.prepare_dataset --config configs/default.yaml   # video+labels -> YOLO crops
python -m balltrack.train          --config configs/default.yaml    # fine-tune YOLO26n
python -m balltrack.predict data/received/part1.mp4 --output prediction.csv [--show]
python -m balltrack.evaluate --pred prediction.csv --gt data/received/part1.csv
python analysis/explore_dataset.py --labels data/received/part1.csv --fps 60   # EDA
```

## Architecture (pipeline, runs in this order)
`prepare_dataset → train → tracker(+predict) → evaluate`. Stages communicate via
files on disk (YOLO dataset, `best.pt`, CSV), not shared memory — each stage is
run and tested independently. Detector = **YOLO26n** (lightweight, small-target
aware). Post-processing = **Kalman filter** (`tracker.py`).

## NON-NEGOTIABLE design decisions — do NOT "simplify" these away
1. **Native-resolution crops.** The ball is ~12px in 1080p. We train AND infer on
   640px crops at native scale so it never shrinks. Never feed a downscaled full
   frame to the detector (that crushes the ball to ~3px). Full-frame reacquisition
   uses `imgsz=1920` precisely to avoid downscaling.
2. **Temporal train/val split, never random.** Adjacent 60fps frames are near
   duplicates; a random split leaks and inflates the score. Split by time + a
   buffer band (`split_buffer`). See `temporal_split()`.
3. **Kalman outlier gate.** Reject detections implying >`max_jump` px/frame
   (kills HUD/radar false positives). Coast only across SHORT gaps; large gaps are
   scene cuts → skip (the brief allows skipping invisible-ball frames).
4. **Honest metrics on the held-out segment only** (frames ≥ val start). Never
   report accuracy on the full video (it includes training frames).

## Data facts (from EDA, drive the configs)
27,664 labels, 96.6% coverage, 60fps, ~8min. 10 gaps (all large = scene cuts).
Ball speed: median 2, p99 16, max 534 (a scene cut) px/frame. Hence `max_jump=40`.

## Conventions
- Prose-style code: typed, docstringed, **no magic numbers** (all tunables live in
  `config.py` dataclasses, values in `configs/default.yaml`).
- **Lazy heavy imports**: `ultralytics`/`torch` are imported INSIDE functions in
  `train.py`/`predict.py` so `prepare_dataset`/`evaluate`/`tracker` stay torch-free.
- Output CSV path is `--output`, default `part1.csv` (spec-exact). Use
  `prediction.csv` during development to avoid clobbering the GT file.
- `data/` and `outputs/` are gitignored (large/generated). Reviewer adds the video.

## Current state (keep this honest and up to date)
- `prepare_dataset.py` — DONE. Pure logic (split, point→box) unit-tested on real CSV.
- `tracker.py` — DONE. Kalman + gate tested offline (rejects injected outlier).
- `evaluate.py` — DONE. Tested on synthetic predictions; val restriction verified.
- `train.py` — WRITTEN, **not yet run end-to-end** (needs GPU + video).
- `predict.py` — WRITTEN, **not yet run end-to-end** (needs `best.pt` + video).
- First real check after `prepare_dataset`: open a few crops in
  `data/yolo/images/train/` and confirm the box sits on the ball.
- NOT built yet: streaming, error-over-time plot in `evaluate.py`.
```