"""
Exploratory Data Analysis — Soccer Ball Position dataset
==============================================================================

WHY THIS FILE EXISTS
--------------------
Before writing a single line of model code, we look at the data. Every design
decision downstream (how we train, how we split, how we post-process, how we
reject false positives) is justified by a measurable property of THIS dataset,
not by a generic recipe. This script reproduces that analysis and prints, for
each finding, the concrete engineering decision it drives.

Run:
    python analysis/explore_dataset.py --labels data/part1.csv --fps 60

Outputs:
    - A narrated report to stdout (numbers + the decision each one drives).
    - Figures saved under analysis/figures/ (coverage timeline, gap sizes,
      speed histogram, position heatmap) for the write-up / presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRAME_W, FRAME_H = 1920, 1080


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"frame_no", "ball_x", "ball_y"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    # Sort by time
    return df.sort_values("frame_no").reset_index(drop=True)


def analyse_coverage(df: pd.DataFrame) -> None:
    section("1. VOLUME & COVERAGE  (how much usable data do we actually have)")
    f = df["frame_no"].values
    span = int(f.max() - f.min() + 1)
    coverage = len(df) / span * 100
    print(f"  labelled frames : {len(df):,}")
    print(f"  frame_no range  : {f.min()} .. {f.max()}  (span {span:,} frames)")
    print(f"  coverage        : {coverage:.1f}% of the span is labelled")


def analyse_gaps(df: pd.DataFrame, fps: int) -> np.ndarray:
    section("2. GAPS  (are the missing frames brief occlusions or scene changes)")
    f = df["frame_no"].values
    step = np.diff(f)
    holes = step[step > 1] - 1  # number of misssing frames inside each gap
    print(f"  number of gaps: {(step > 1).sum()}")
    print(f"  total missing frames: {int(holes.sum())}")
    if len(holes):
        print(f"  largest gap: {int(holes.max())} frames "
              f"(~{holes.max()/fps:.1f}s)")

    return holes


def analyse_speed(df: pd.DataFrame) -> None:
    section("3. BALL SPEED  (how fast does the ball move frame-to-frame)")
    f = df["frame_no"].values
    x = df["ball_x"].values.astype(float)
    y = df["ball_y"].values.astype(float)
    consecutive = np.diff(f) == 1  # speed only valid between adjacent frames
    dx = np.diff(x)[consecutive]
    dy = np.diff(y)[consecutive]
    speed = np.sqrt(dx**2 + dy**2)
    for name, val in [
        ("median", np.median(speed)), ("mean", speed.mean()),
        ("p95", np.percentile(speed, 95)), ("p99", np.percentile(speed, 99)),
        ("max", speed.max()),
    ]:
        print(f"  {name:>6} : {val:6.1f} px/frame")


def analyse_positions(df: pd.DataFrame, out_dir: Path) -> None:
    section("4. POSITION DISTRIBUTION  (any suspicious clusters? out-of-frame)")
    x = df["ball_x"].values
    y = df["ball_y"].values
    oob = int(((x < 0) | (x > FRAME_W) | (y < 0) | (y > FRAME_H)).sum())
    print(f"  x range : {x.min()} .. {x.max()}")
    print(f"  y range : {y.min()} .. {y.max()}")
    print(f"  points outside the 1920x1080 frame : {oob}")


def make_figures(df: pd.DataFrame, holes: np.ndarray, out_dir: Path) -> None:
    """Save figures for the write-up. Visuals make the reasoning legible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = df["frame_no"].values
    x = df["ball_x"].values.astype(float)
    y = df["ball_y"].values.astype(float)

    # 4a. Coverage timeline: where are the labelled frames vs the gaps.
    present = np.zeros(int(f.max()) + 1, dtype=np.uint8)
    present[f] = 1
    fig, ax = plt.subplots(figsize=(11, 1.6))
    ax.imshow(present[None, :], aspect="auto", cmap="Greens",
              extent=[0, len(present), 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("frame number")
    ax.set_title("Label coverage timeline (green = labelled, white = gap)")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_timeline.png", dpi=130)
    plt.close(fig)

    # 4b. Speed histogram (consecutive frames only), clipped for readability.
    consecutive = np.diff(f) == 1
    speed = np.sqrt(np.diff(x)[consecutive] ** 2 + np.diff(y)[consecutive] ** 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.clip(speed, 0, 40), bins=60, color="#3b7dd8")
    ax.axvline(np.percentile(speed, 99), color="crimson", ls="--",
               label=f"p99 = {np.percentile(speed,99):.0f}px")
    ax.set_xlabel("ball speed (px/frame, clipped at 40)")
    ax.set_ylabel("count")
    ax.set_title("Frame-to-frame ball speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "speed_hist.png", dpi=130)
    plt.close(fig)

    # 4c. Position heatmap: spot any off-pitch hotspots (e.g. radar/HUD).
    fig, ax = plt.subplots(figsize=(6, 3.6))
    h = ax.hist2d(x, y, bins=[64, 36], range=[[0, FRAME_W], [0, FRAME_H]],
                  cmap="magma")
    ax.invert_yaxis()  # image coords: y grows downward
    ax.set_xlabel("ball_x"); ax.set_ylabel("ball_y")
    ax.set_title("Ball position density (image coordinates)")
    fig.colorbar(h[3], ax=ax, label="frames")
    fig.tight_layout()
    fig.savefig(out_dir / "position_heatmap.png", dpi=130)
    plt.close(fig)

    print(f"\n  figures saved to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/part1.csv"))
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--figdir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()

    df = load_labels(args.labels)
    analyse_coverage(df)
    
    holes = analyse_gaps(df, args.fps)
    
    analyse_speed(df)
    
    analyse_positions(df, args.figdir)
    make_figures(df, holes, args.figdir)

if __name__ == "__main__":
    main()