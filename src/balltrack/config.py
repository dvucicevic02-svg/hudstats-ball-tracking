"""Central configuration. Defaults are derived from the EDA, not guessed.

Keeping every tunable in one typed place (instead of scattered magic numbers)
is the difference between a script and something maintainable. Values can be
overridden from a YAML file via `Config.from_yaml(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # --- inputs ---
    video: Path = Path("data/part1.mp4")
    labels: Path = Path("data/part1.csv")
    out_dir: Path = Path("data/yolo")  # generated YOLO-format dataset

    # --- video geometry (brief says 1080p) ---
    frame_w: int = 1920
    frame_h: int = 1080

    # --- sampling (EDA: data is dense -> drop redundant near-identical frames) ---
    subsample: int = 6  # keep every Nth labelled frame for training

    # --- point -> box (labels are the ball CENTRE; YOLO needs boxes) ---
    box_size: int = 24  # square side in px around the centre; ~ball diameter

    # --- crops (KEY DECISION: train at native scale so the ball never shrinks) ---
    crop_size: int = 640        # square crop side in px, matches train imgsz
    jitter: float = 0.35        # max offset of ball from crop centre, as a fraction
    hard_negative_ratio: float = 0.25  # background-only crops (HUD/crowd) per positive

    # --- temporal split (EDA: random split leaks; adjacent frames ~identical) ---
    val_fraction: float = 0.2   # last 20% of the timeline is held out
    split_buffer: int = 300     # frames dropped between train and val (no leakage)

    seed: int = 1337

    def __post_init__(self) -> None:
        # YAML gives plain strings; force path fields to Path so `/` joins work
        # regardless of whether the value came from code or from default.yaml.
        self.video = Path(self.video)
        self.labels = Path(self.labels)
        self.out_dir = Path(self.out_dir)


@dataclass
class TrainConfig:
    model: str = "yolo26n.pt"  # lightweight; STAL/ProgLoss help small targets
    imgsz: int = 640
    epochs: int = 75
    batch: int = 16            # comfortable on a 6GB RTX 2060 at 640px
    patience: int = 15         # early stopping
    device: int | str = 0      # GPU 0; set "cpu" to force CPU


@dataclass
class TrackConfig:
    # EDA-derived: real motion ~2-3 px/frame, p99 ~16 px/frame.
    # NOTE: the ROI crop size is NOT defined here — inference reuses
    # data.crop_size (single source of truth) so train/infer scale can't diverge.
    reacquire_imgsz: int = 1920    # full-frame scan = no downscaling -> ball stays sharp
    max_jump: float = 40.0         # outlier gate (px/frame); well above p99 -> kills HUD hits
    max_coast: int = 8             # Kalman may bridge gaps up to this many frames
    conf: float = 0.25             # detection confidence threshold


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    track: TrackConfig = field(default_factory=TrackConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            track=TrackConfig(**raw.get("track", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)