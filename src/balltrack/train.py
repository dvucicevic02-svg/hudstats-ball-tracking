"""
train.py — fine-tune YOLO26 on the native-resolution ball crops.

We adapt a COCO-pretrained model to one class: `ball`. The variant is set by
train.size in the config (n | s | m | l | x), so the same code trains any size
and each run is named yolo26<size>_ball. Training reads the dataset built by
prepare_dataset.py (which already did the temporal train/val split), so
Ultralytics' own val metrics here are computed on the held-out segment, not on
memorised neighbours.

Run:
    python -m balltrack.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from balltrack.config import Config
from balltrack.tracking import setup_mlflow


def train(cfg: Config) -> Path:
    # Imported lazily so that `prepare_dataset` / `evaluate` don't pull in torch.
    from ultralytics import YOLO
    from ultralytics import settings as ul_settings

    dataset_yaml = cfg.data.out_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} not found. Run `python -m balltrack.prepare_dataset` first."
        )

    tracking_uri = setup_mlflow(cfg.mlflow.experiment)
    ul_settings.update({"mlflow": tracking_uri is not None})

    model = YOLO(cfg.train.model)  # yolo26{size}.pt (downloads pretrained weights)

    results = model.train(
        data=str(dataset_yaml),
        imgsz=cfg.train.imgsz,
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        patience=cfg.train.patience,   # guards against pixel-memorising
        device=cfg.train.device,
        amp=True,          # mixed precision (float16 insted of float32) -> faster, less VRAM on the 6GB 2060
        project="outputs/train",
        name=cfg.train.run_name,   # yolo26{size}_ball -> per-variant folder + MLflow run
        exist_ok=True,
        mosaic=1.0,        # stitch 4 images per sample -> ball seen in varied contexts/positions
        scale=0.5,         # random zoom +/-50% -> robust to ball scale changes
        translate=0.1,     # random shift up to 10% -> ball not always centred
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, # HSV colour jitter: hue off (ball is white, keep its colour)
                                         # small saturation/value range to simulate different pitch lighting.
        fliplr=0.5,        # horizontal flip 50%: pitch is left-right symmetric
        flipud=0.0,        # no vertical flip: a flipped pitch is unnatural
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining done. Best weights: {best}")

    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO26 on the ball crops.")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    train(cfg)


if __name__ == "__main__":
    main()