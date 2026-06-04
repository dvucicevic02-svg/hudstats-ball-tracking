"""
train.py — fine-tune YOLO26n on the native-resolution ball crops.

We adapt a COCO-pretrained nano model ('lightweight', as the brief asks) to one
class: `ball`. Training reads the dataset built by prepare_dataset.py (which
already did the temporal train/val split), so Ultralytics' own val metrics here
are computed on the held-out segment — not on memorised neighbours.

Run:
    python -m balltrack.train --config configs/default.yaml

Notes
  * AMP (mixed precision) is on by default in Ultralytics and is ideal for the
    RTX 2060's Tensor cores — faster and lighter on the 6GB VRAM.
  * MLflow logging is always on: hyper-params, metrics and the best weights are
    logged to the tracking server (MLFLOW_TRACKING_URI, default
    http://127.0.0.1:5000). If the server is down it degrades to a warning.
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

    # Point MLflow at the tracking server BEFORE the ultralytics callback runs:
    # it reads MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME from the environment,
    # which setup_mlflow() has just exported. Returns None if unavailable.
    tracking_uri = setup_mlflow(cfg.mlflow.experiment)

    # Ultralytics has native MLflow integration; enable it (it then honours the
    # URI/experiment set above). Off only when the server/library is unavailable.
    ul_settings.update({"mlflow": tracking_uri is not None})

    model = YOLO(cfg.train.model)  # e.g. yolo26n.pt (downloads pretrained weights)

    results = model.train(
        data=str(dataset_yaml),
        imgsz=cfg.train.imgsz,
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        patience=cfg.train.patience,   # early stopping: guards against pixel-memorising
        device=cfg.train.device,
        amp=True,
        project="outputs/train",
        name="yolo26n_ball",
        exist_ok=True,
        # --- augmentation: teach ball APPEARANCE, not exact frames ---
        # mosaic/scale/translate add variety; we keep colour jitter mild because
        # the FIFA domain is visually stable (same pitch, lighting, ball).
        mosaic=1.0,
        scale=0.5,
        translate=0.1,
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
        fliplr=0.5,
        flipud=0.0,           # a flipped pitch is unnatural; keep vertical fixed
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining done. Best weights: {best}")
    print("Ultralytics val metrics (on the temporal hold-out) are in the run dir; "
          "for pixel-error on the full video pipeline, run balltrack.evaluate.")

    # The ultralytics callback already uploads the run artifacts, but we log
    # best.pt explicitly so the deployable weight is guaranteed to be attached
    # under a predictable path. Guarded so a tracking outage can't fail training.
    if tracking_uri is not None and best.exists():
        try:
            import mlflow

            with mlflow.start_run(run_name="best-weights"):
                mlflow.log_artifact(str(best), artifact_path="weights")
            print(f"Logged best.pt to MLflow ({tracking_uri}).")
        except Exception as exc:
            print(f"(could not log best.pt to MLflow: {exc}; weights saved locally)")

    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO26n on the ball crops.")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    train(cfg)


if __name__ == "__main__":
    main()