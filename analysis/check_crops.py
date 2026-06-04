"""
check_crops.py — verify the prepared dataset by EYE.

prepare_dataset.py writes crops (images) + YOLO labels (normalised 0-1 numbers).
Those numbers are unreadable on their own. This script does the reverse of label
creation: it reads each `.txt`, converts the normalised box back to pixels on its
crop, and draws it — exactly like show_ball_dataset.py does on the full video,
but adapted to our 640px crops and their own coordinate system.

Open the saved images: the green box MUST sit on the ball. Hard-negative crops
(empty .txt) get no box and should contain NO ball — that's correct.

RUN
    python analysis/check_crops.py --dataset data/yolo --split train --n 16
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


def draw_label(img, line: str):
    """Draw one YOLO box (class cx cy w h, all normalised) onto the image."""
    h, w = img.shape[:2]
    _, cx, cy, bw, bh = (float(v) for v in line.split())
    x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # also mark the centre, like the ground-truth viewer
    cv2.drawMarker(img, (int(cx * w), int(cy * h)), (0, 255, 0),
                   cv2.MARKER_CROSS, 12, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eyeball-check prepared crops.")
    ap.add_argument("--dataset", type=Path, default=Path("data/yolo"))
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--n", type=int, default=16, help="How many crops to render.")
    ap.add_argument("--out", type=Path, default=Path("analysis/check_crops"))
    args = ap.parse_args()

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    args.out.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}. Run prepare_dataset first.")
    random.seed(0)
    picks = random.sample(images, min(args.n, len(images)))

    pos, neg = 0, 0
    for p in picks:
        img = cv2.imread(str(p))
        lbl = (lbl_dir / f"{p.stem}.txt")
        lines = lbl.read_text().strip().splitlines() if lbl.exists() else []
        for line in lines:
            if line.strip():
                draw_label(img, line)
        tag = "POS" if lines and lines[0].strip() else "NEG"
        pos += tag == "POS"; neg += tag == "NEG"
        cv2.imwrite(str(args.out / f"{tag}_{p.stem}.png"), img)

    print(f"Rendered {len(picks)} crops -> {args.out}/  (positives={pos}, negatives={neg})")
    print("Open them: the GREEN box must sit on the ball. NEG_* must have NO ball.")


if __name__ == "__main__":
    main()