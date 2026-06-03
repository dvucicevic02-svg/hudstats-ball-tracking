"""
prepare_dataset.py — turn the raw video + centre-point labels into a YOLO dataset.

PIPELINE
    1. Load & validate labels.
    2. SUBSAMPLE (data is dense; near-identical frames add nothing).
    3. TEMPORAL split into train/val (a random split would leak — see below).
    4. For each kept frame: cut a native-resolution CROP around the ball
       (the ball must NOT be downscaled), convert the centre point to a YOLO
       box, and write image + label. Plus a fraction of HARD-NEGATIVE crops
       (background only) so the model learns what is NOT a ball.
    5. Emit dataset.yaml for Ultralytics.

WHY A TEMPORAL SPLIT
    At 60fps the ball moves ~2px between frames, so frame t and t+1 are almost
    the same image. A random train/val split puts near-duplicates on both sides,
    so the val score measures memorisation, not generalisation, and collapses on
    the held-out `part2`. We therefore hold out a CONTIGUOUS block at the end of
    the timeline, with a buffer band so train and val never touch.

WHY SEQUENTIAL READING (not cap.set per frame)
    Seeking by frame index (CAP_PROP_POS_FRAMES) is unreliable on many codecs —
    it snaps to the nearest keyframe, so the decoded frame may not be the one the
    label refers to. For DATASET generation that silently corrupts labels. We
    decode the video once, in order, and act on frames as we pass them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from balltrack.config import Config, DataConfig
from balltrack.crop import crop_origin


# --------------------------------------------------------------------------- #
# Pure logic (no video, no I/O) — easy to unit-test.
# --------------------------------------------------------------------------- #
def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"frame_no", "ball_x", "ball_y"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")
    return df.sort_values("frame_no").reset_index(drop=True)


def temporal_split(
    frames: np.ndarray, val_fraction: float, buffer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the LAST `val_fraction` of the timeline as validation.

    A `buffer` band of frames just before the split is dropped from BOTH sets so
    that no validation frame has a near-identical neighbour in training.
    """
    frames = np.sort(frames)
    cut_idx = int(len(frames) * (1.0 - val_fraction))
    split_frame = frames[cut_idx]
    val = frames[frames >= split_frame]
    train = frames[frames < (split_frame - buffer)]
    return train, val


def point_to_yolo_box(
    cx: float, cy: float, box: int,
    crop_x0: int, crop_y0: int, crop_w: int, crop_h: int,
) -> tuple[float, float, float, float]:
    """Centre point -> YOLO (xc, yc, w, h), normalised to the CROP, clamped."""
    local_cx = (cx - crop_x0) / crop_w
    local_cy = (cy - crop_y0) / crop_h
    w = box / crop_w
    h = box / crop_h
    # clamp so a ball near the crop edge stays a valid (in-bounds) box
    local_cx = float(np.clip(local_cx, w / 2, 1 - w / 2))
    local_cy = float(np.clip(local_cy, h / 2, 1 - h / 2))
    return local_cx, local_cy, w, h


def crop_window(
    cx: float, cy: float, crop: int, jitter: float,
    frame_w: int, frame_h: int, rng: np.random.Generator,
) -> tuple[int, int]:
    """Top-left of a `crop`x`crop` training window around (cx, cy).

    Jitter (a TRAINING-only augmentation) pushes the ball off-centre so the model
    doesn't learn "ball == middle". The actual window geometry then comes from the
    SHARED `crop_origin`, so training and inference cannot diverge in scale.
    """
    max_off = jitter * crop / 2
    jx = cx + rng.uniform(-max_off, max_off)
    jy = cy + rng.uniform(-max_off, max_off)
    return crop_origin(jx, jy, crop, frame_w, frame_h)


def negative_window(
    cx: float, cy: float, crop: int, frame_w: int, frame_h: int,
    rng: np.random.Generator, min_clear: int = 80, tries: int = 20,
) -> tuple[int, int] | None:
    """A random crop that does NOT contain the ball -> a hard negative."""
    for _ in range(tries):
        x0 = int(rng.integers(0, max(1, frame_w - crop)))
        y0 = int(rng.integers(0, max(1, frame_h - crop)))
        inside = (x0 - min_clear <= cx <= x0 + crop + min_clear and
                  y0 - min_clear <= cy <= y0 + crop + min_clear)
        if not inside:
            return x0, y0
    return None


# --------------------------------------------------------------------------- #
# Dataset generation (touches the video).
# --------------------------------------------------------------------------- #
def _write_example(
    img_dir: Path, lbl_dir: Path, name: str, crop: np.ndarray,
    label: tuple[float, float, float, float] | None,
) -> None:
    cv2.imwrite(str(img_dir / f"{name}.jpg"), crop)
    # YOLO convention: a negative is an image with an empty (or absent) label file.
    text = "" if label is None else "0 %.6f %.6f %.6f %.6f\n" % label
    (lbl_dir / f"{name}.txt").write_text(text)


def generate_dataset(cfg: DataConfig) -> None:
    df = load_labels(cfg.labels)
    rng = np.random.default_rng(cfg.seed)

    # Subsample, then temporally split (order matters: split the kept frames).
    kept = df["frame_no"].values[:: cfg.subsample]
    train_f, val_f = temporal_split(kept, cfg.val_fraction, cfg.split_buffer)
    split_of = {int(f): "train" for f in train_f}
    split_of.update({int(f): "val" for f in val_f})
    pos = df.set_index("frame_no")[["ball_x", "ball_y"]].to_dict("index")

    # Output tree: data/yolo/{images,labels}/{train,val}
    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = cfg.out_dir / kind / split
            d.mkdir(parents=True, exist_ok=True)
            dirs[(kind, split)] = d

    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video}")

    targets = set(split_of)
    written = {"train": 0, "val": 0, "neg": 0}
    idx = -1
    while targets:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx not in targets:
            continue
        targets.discard(idx)

        split = split_of[idx]
        cx, cy = pos[idx]["ball_x"], pos[idx]["ball_y"]

        # Positive crop (native resolution -> ball keeps its real size).
        x0, y0 = crop_window(cx, cy, cfg.crop_size, cfg.jitter,
                             cfg.frame_w, cfg.frame_h, rng)
        crop = frame[y0:y0 + cfg.crop_size, x0:x0 + cfg.crop_size]
        box = point_to_yolo_box(cx, cy, cfg.box_size, x0, y0,
                                cfg.crop_size, cfg.crop_size)
        _write_example(dirs[("images", split)], dirs[("labels", split)],
                       f"f{idx:06d}", crop, box)
        written[split] += 1

        # Hard negative (background only) for a fraction of frames.
        if rng.random() < cfg.hard_negative_ratio:
            nw = negative_window(cx, cy, cfg.crop_size, cfg.frame_w, cfg.frame_h, rng)
            if nw is not None:
                nx, ny = nw
                neg = frame[ny:ny + cfg.crop_size, nx:nx + cfg.crop_size]
                _write_example(dirs[("images", split)], dirs[("labels", split)],
                               f"f{idx:06d}_neg", neg, None)
                written["neg"] += 1

    cap.release()
    _write_dataset_yaml(cfg)
    print(f"Done. train={written['train']} val={written['val']} "
          f"negatives={written['neg']}  ->  {cfg.out_dir}")


def _write_dataset_yaml(cfg: DataConfig) -> None:
    yaml_text = (
        f"path: {cfg.out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: ball\n"
    )
    (cfg.out_dir / "dataset.yaml").write_text(yaml_text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the YOLO dataset from video + labels.")
    ap.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.video:   cfg.data.video = args.video
    if args.labels:  cfg.data.labels = args.labels
    if args.out_dir: cfg.data.out_dir = args.out_dir
    generate_dataset(cfg.data)


if __name__ == "__main__":
    main()