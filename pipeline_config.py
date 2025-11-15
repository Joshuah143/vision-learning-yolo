from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILENAME = "pipeline_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / CONFIG_FILENAME


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass(frozen=True, slots=True)
class PlatesConfig:
    num_plates: int
    output_dir: Path
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetSectionConfig:
    dataset_root: Path
    plates_dir: Path
    backgrounds_dir: Path
    image_height: int
    image_width: int
    train_ratio: float
    augmentations_per_plate: int
    fizz_sign_class_index: int
    class_names: tuple[str, ...]
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    weights_path: Path
    project: str
    run_name: str
    epochs: int
    batch: int
    imgsz: int
    device: str
    optimizer: str
    lr0: float
    patience: int


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    plates: PlatesConfig
    dataset: DatasetSectionConfig
    training: TrainingConfig


_CONFIG_CACHE: PipelineConfig | None = None


def load_pipeline_config(config_path: Path | None = None) -> PipelineConfig:
    """Load the pipeline configuration from YAML."""
    global _CONFIG_CACHE
    if config_path is None and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found at {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a mapping at the top level.")

    base_dir = path.parent

    plates_raw: dict[str, Any] = data.get("plates", {})
    dataset_raw: dict[str, Any] = data.get("dataset", {})
    training_raw: dict[str, Any] = data.get("training", {})

    plates_cfg = PlatesConfig(
        num_plates=int(plates_raw.get("num_plates", 0)),
        output_dir=_resolve_path(base_dir, plates_raw.get("output_dir", "plates_unmodified")),
        seed=plates_raw.get("seed"),
    )

    class_names = dataset_raw.get("class_names")
    if not class_names:
        raise ValueError("dataset.class_names must be provided in pipeline config.")

    dataset_cfg = DatasetSectionConfig(
        dataset_root=_resolve_path(base_dir, dataset_raw.get("dataset_root", "fizz_yolo_dataset")),
        plates_dir=_resolve_path(base_dir, dataset_raw.get("plates_dir", "plates_unmodified")),
        backgrounds_dir=_resolve_path(base_dir, dataset_raw.get("backgrounds_dir", "backgrounds")),
        image_height=int(dataset_raw.get("image_height", 720)),
        image_width=int(dataset_raw.get("image_width", 1280)),
        train_ratio=float(dataset_raw.get("train_ratio", 0.8)),
        augmentations_per_plate=int(dataset_raw.get("augmentations_per_plate", 20)),
        fizz_sign_class_index=int(dataset_raw.get("fizz_sign_class_index", 3)),
        class_names=tuple(str(name) for name in class_names),
        seed=dataset_raw.get("seed"),
    )

    training_cfg = TrainingConfig(
        weights_path=_resolve_path(base_dir, training_raw.get("weights_path", "yolo12l.pt")),
        project=str(training_raw.get("project", "enph353-fizz-yolo")),
        run_name=str(training_raw.get("run_name", "yolo_fizz_obstacles_v1")),
        epochs=int(training_raw.get("epochs", 10)),
        batch=int(training_raw.get("batch", 128)),
        imgsz=int(training_raw.get("imgsz", 640)),
        device=str(training_raw.get("device", "mps")),
        optimizer=str(training_raw.get("optimizer", "AdamW")),
        lr0=float(training_raw.get("lr0", 1e-3)),
        patience=int(training_raw.get("patience", 20)),
    )

    config = PipelineConfig(
        plates=plates_cfg,
        dataset=dataset_cfg,
        training=training_cfg,
    )

    if config_path is None:
        _CONFIG_CACHE = config

    return config

