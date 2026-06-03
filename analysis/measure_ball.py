"""
measure_ball.py — measure the ball's pixel size, instead of guessing it.

WHY
    The training box size (config: box_size) should match the real ball, but the
    labels give only the ball CENTRE, not its size. This tool measures the ball
    on a sample of frames two ways:
      * AUTO   — around each labelled centre, find the bright ball blob and report
                 its diameter. Robust-ish; we report the MEDIAN over many frames.
      * VISUAL — save a zoomed crop around the centre with a pixel grid overlaid,
                 so you can confirm the diameter with your own eyes (the reliable
                 ground truth, especially when the ball is on a white line).

    Use the median AUTO value as a starting point, confirm against a few VISUAL
    crops, then set `box_size` in configs/default.yaml to roughly 1.8-2.2x the
    measured diameter (a little margin around the ball).

RUN
    python analysis/measure_ball.py \
        --video data/received/part1.mp4 --labels data/received/part1.csv \
        --samples 40 --out analysis/ball_size

OUTPUT
    * prints per-frame diameters and the median to stdout
    * saves zoomed grid crops to the --out folder for visual confirmation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def auto_diameter(gray_patch: np.ndarray) -> float | None:
    """Estimate ball diameter (px) in a small grayscale patch centred on the ball.

    The ball is brighter than the pitch, so we threshold the patch and measure
    the connected bright blob nearest the centre. Returns None if nothing clear.
    """
    # Otsu picks a bright/dark cutoff automatically for this patch.
    _, thresh = cv2.threshold(gray_patch, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if n <= 1:
        return None
    cx, cy = gray_patch.shape[1] / 2, gray_patch.shape[0] / 2
    best, best_d = None, 1e9
    for i in range(1, n):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4 or area > gray_patch.size * 0.5:  # ignore noise / huge blobs
            continue
        d = np.hypot(centroids[i][0] - cx, centroids[i][1] - cy)
        if d < best_d:
            best_d, best = d, i
    if best is None:
        return None
    # Diameter ~ average of the blob's width and height in pixels.
    w = stats[best, cv2.CC_STAT_WIDTH]
    h = stats[best, cv2.CC_STAT_HEIGHT]
    return float((w + h) / 2)


def save_grid_crop(frame: np.ndarray, cx: int, cy: int, out: Path,
                   half: int = 20, zoom: int = 12) -> None:
    """Save a zoomed crop around (cx, cy) with a 1px grid + centre marker."""
    H, W = frame.shape[:2]
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    patch = frame[y0:y1, x0:x1]
    big = cv2.resize(patch, (patch.shape[1] * zoom, patch.shape[0] * zoom),
                     interpolation=cv2.INTER_NEAREST)  # NEAREST keeps pixels crisp
    # grid every original pixel
    for gx in range(0, big.shape[1], zoom):
        cv2.line(big, (gx, 0), (gx, big.shape[0]), (60, 60, 60), 1)
    for gy in range(0, big.shape[0], zoom):
        cv2.line(big, (0, gy), (big.shape[1], gy), (60, 60, 60), 1)
    # mark the labelled centre
    mcx, mcy = (cx - x0) * zoom + zoom // 2, (cy - y0) * zoom + zoom // 2
    cv2.drawMarker(big, (mcx, mcy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(str(out), big)


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the ball's pixel size.")
    ap.add_argument("--video", type=Path, default=Path("data/received/part1.mp4"))
    ap.add_argument("--labels", type=Path, default=Path("data/received/part1.csv"))
    ap.add_argument("--samples", type=int, default=40,
                    help="How many labelled frames to sample across the video.")
    ap.add_argument("--out", type=Path, default=Path("analysis/ball_size"))
    ap.add_argument("--patch", type=int, default=40,
                    help="Half-size (px) of the patch searched around the centre.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels).sort_values("frame_no").reset_index(drop=True)
    # Evenly spaced sample across the whole timeline (not just the start).
    idx = np.linspace(0, len(df) - 1, args.samples).astype(int)
    sample = df.iloc[idx]
    wanted = {int(r.frame_no): (int(r.ball_x), int(r.ball_y)) for r in sample.itertuples()}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    diameters: list[float] = []
    saved = 0
    fno = -1
    targets = set(wanted)
    while targets:
        ret, frame = cap.read()
        if not ret:
            break
        fno += 1
        if fno not in targets:
            continue
        targets.discard(fno)
        cx, cy = wanted[fno]

        # AUTO measurement on a grayscale patch around the centre.
        H, W = frame.shape[:2]
        x0, y0 = max(0, cx - args.patch), max(0, cy - args.patch)
        patch = cv2.cvtColor(frame[y0:min(H, cy + args.patch),
                                   x0:min(W, cx + args.patch)], cv2.COLOR_BGR2GRAY)
        d = auto_diameter(patch)
        if d is not None:
            diameters.append(d)
            print(f"frame {fno:>6}: ~{d:.1f} px")

        # VISUAL crop for the first ~12 samples (enough to eyeball).
        if saved < 12:
            save_grid_crop(frame, cx, cy, args.out / f"ball_f{fno:06d}.png")
            saved += 1

    cap.release()

    if diameters:
        med = float(np.median(diameters))
        print("\n" + "=" * 50)
        print(f"AUTO median diameter : {med:.1f} px  (n={len(diameters)})")
        print(f"suggested box_size   : {round(med * 2)} px  (~2x diameter)")
        print(f"visual crops saved   : {args.out}/  — confirm by eye")
        print("=" * 50)
    else:
        print("\nNo automatic measurement succeeded — rely on the visual crops "
              f"in {args.out}/ (count the ball's pixels on the grid).")


if __name__ == "__main__":
    main()