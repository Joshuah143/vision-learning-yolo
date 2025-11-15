#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import modal

from gen_dataset import dataset_config_from_settings, generate_dataset
from gen_plates import generate_plates
from pipeline_config import (
    DatasetSectionConfig,
    PipelineConfig,
    PlatesConfig,
    TrainingConfig,
    load_pipeline_config,
)
from train import run_training

app = modal.App("enph353-h100-pipeline")

HERE = Path(__file__).resolve().parent
VOLUME_ROOT = Path("/vol/enph353")
ARTIFACTS_DIR = VOLUME_ROOT / "artifacts"


def _volume_path_for(path: Path) -> Path:
    try:
        relative = path.relative_to(HERE)
    except ValueError:
        relative = Path(path.name)
    return VOLUME_ROOT / relative


DATA_VOLUME = modal.Volume.from_name("enph353-training-data")

BASE_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .env({"PYTHONPATH": "/app"})
    .uv_sync()
    .add_local_file("pipeline_config.yaml", "/app/pipeline_config.yaml")
    .add_local_file("gen_dataset.py", "/app/gen_dataset.py")
    .add_local_file("gen_plates.py", "/app/gen_plates.py")
    .add_local_file("pipeline_config.py", "/app/pipeline_config.py")
    .add_local_file("train.py", "/app/train.py")
    .add_local_dir("gen_tools", "/app/gen_tools")
)


def _ensure_volume_layout(plates_dir: Path, dataset_root: Path) -> None:
    for path in (plates_dir, dataset_root, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _plates_config_for_volume(plates_cfg: PlatesConfig) -> PlatesConfig:
    return replace(plates_cfg, output_dir=_volume_path_for(plates_cfg.output_dir))


def _dataset_config_for_volume(dataset_cfg: DatasetSectionConfig) -> DatasetSectionConfig:
    return replace(
        dataset_cfg,
        dataset_root=_volume_path_for(dataset_cfg.dataset_root),
        plates_dir=_volume_path_for(dataset_cfg.plates_dir),
    )


def _training_config_for_gpu(training_cfg: TrainingConfig) -> TrainingConfig:
    if training_cfg.device.lower() == "cuda":
        return training_cfg
    return replace(training_cfg, device="cuda")


def _path_from_str(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (HERE / path).resolve()
    return path


def _config_from_payload(payload: dict[str, Any] | None) -> PipelineConfig:
    if payload is None:
        return load_pipeline_config()

    plates_raw = payload["plates"]
    plates_cfg = PlatesConfig(
        num_plates=int(plates_raw["num_plates"]),
        output_dir=_path_from_str(plates_raw["output_dir"]),
        seed=plates_raw.get("seed"),
    )

    dataset_raw = payload["dataset"]
    dataset_cfg = DatasetSectionConfig(
        dataset_root=_path_from_str(dataset_raw["dataset_root"]),
        plates_dir=_path_from_str(dataset_raw["plates_dir"]),
        backgrounds_dir=_path_from_str(dataset_raw["backgrounds_dir"]),
        image_height=int(dataset_raw["image_height"]),
        image_width=int(dataset_raw["image_width"]),
        train_ratio=float(dataset_raw["train_ratio"]),
        augmentations_per_plate=int(dataset_raw["augmentations_per_plate"]),
        fizz_sign_class_index=int(dataset_raw["fizz_sign_class_index"]),
        class_names=tuple(str(name) for name in dataset_raw["class_names"]),
        seed=dataset_raw.get("seed"),
    )

    training_raw = payload["training"]
    training_cfg = TrainingConfig(
        weights_path=_path_from_str(training_raw["weights_path"]),
        project=str(training_raw["project"]),
        run_name=str(training_raw["run_name"]),
        epochs=int(training_raw["epochs"]),
        batch=int(training_raw["batch"]),
        imgsz=int(training_raw["imgsz"]),
        device=str(training_raw["device"]),
        optimizer=str(training_raw["optimizer"]),
        lr0=float(training_raw["lr0"]),
        patience=int(training_raw["patience"]),
    )

    return PipelineConfig(plates=plates_cfg, dataset=dataset_cfg, training=training_cfg)


def _serialize_config(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "plates": {
            "num_plates": cfg.plates.num_plates,
            "output_dir": str(cfg.plates.output_dir),
            "seed": cfg.plates.seed,
        },
        "dataset": {
            "dataset_root": str(cfg.dataset.dataset_root),
            "plates_dir": str(cfg.dataset.plates_dir),
            "backgrounds_dir": str(cfg.dataset.backgrounds_dir),
            "image_height": cfg.dataset.image_height,
            "image_width": cfg.dataset.image_width,
            "train_ratio": cfg.dataset.train_ratio,
            "augmentations_per_plate": cfg.dataset.augmentations_per_plate,
            "fizz_sign_class_index": cfg.dataset.fizz_sign_class_index,
            "class_names": list(cfg.dataset.class_names),
            "seed": cfg.dataset.seed,
        },
        "training": {
            "weights_path": str(cfg.training.weights_path),
            "project": cfg.training.project,
            "run_name": cfg.training.run_name,
            "epochs": cfg.training.epochs,
            "batch": cfg.training.batch,
            "imgsz": cfg.training.imgsz,
            "device": cfg.training.device,
            "optimizer": cfg.training.optimizer,
            "lr0": cfg.training.lr0,
            "patience": cfg.training.patience,
        },
    }


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": DATA_VOLUME},
    cpu=(4, 16),
    timeout=60 * 30,
)
def generate_plates_remote(pipeline_cfg=None) -> None:
    """Render raw clue plates + metadata into the shared volume."""
    cfg = _config_from_payload(pipeline_cfg)
    plates_cfg = _plates_config_for_volume(cfg.plates)
    dataset_cfg = _dataset_config_for_volume(cfg.dataset)
    _ensure_volume_layout(plates_cfg.output_dir, dataset_cfg.dataset_root)
    generate_plates(plates_cfg)


@app.function(
    image=BASE_IMAGE,
    volumes={"/vol": DATA_VOLUME},
    cpu=32,
    timeout=60 * 60,
)
def build_dataset_remote(pipeline_cfg=None) -> None:
    """Augment plates into YOLO-style dataset stored on the volume."""
    # Configure logging to ensure we see output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting build_dataset_remote")
    
    cfg = _config_from_payload(pipeline_cfg)
    dataset_cfg = _dataset_config_for_volume(cfg.dataset)
    logger.info("Config loaded: plates_dir=%s, dataset_root=%s", 
                dataset_cfg.plates_dir, dataset_cfg.dataset_root)
    
    _ensure_volume_layout(dataset_cfg.plates_dir, dataset_cfg.dataset_root)
    logger.info("Volume layout ensured")
    
    # Check if plates directory exists and has files
    if not dataset_cfg.plates_dir.exists():
        raise FileNotFoundError(
            f"Plates directory does not exist: {dataset_cfg.plates_dir}. "
            "Run generate_plates_remote first."
        )
    
    plate_files = list(dataset_cfg.plates_dir.glob("*.png")) + list(dataset_cfg.plates_dir.glob("*.jpg"))
    logger.info("Found %d plate image files in %s", len(plate_files), dataset_cfg.plates_dir)
    
    if not plate_files:
        raise ValueError(
            f"No plate images found in {dataset_cfg.plates_dir}. "
            "Run generate_plates_remote first."
        )
    
    dataset_config = dataset_config_from_settings(dataset_cfg)
    
    # Set CPU count in environment for generate_dataset to use
    # Modal allocates 32 CPUs, but os.cpu_count() might not reflect this
    detected_cpus = os.cpu_count() or 4
    logger.info("Detected %d CPUs via os.cpu_count(), Modal allocated 32 CPUs", detected_cpus)
    # Set environment variable so generate_dataset can use the allocated CPU count
    os.environ["MODAL_CPU_COUNT"] = "32"
    
    logger.info("Calling generate_dataset")
    generate_dataset(dataset_config, seed=dataset_cfg.seed)
    logger.info("Dataset generation completed")


@app.function(
    image=BASE_IMAGE,
    gpu="H100",
    volumes={"/vol": DATA_VOLUME},
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_remote(pipeline_cfg=None) -> Path:
    """Run YOLO training on an H100 GPU using the dataset stored on the volume."""
    cfg = _config_from_payload(pipeline_cfg)
    dataset_cfg = _dataset_config_for_volume(cfg.dataset)
    _ensure_volume_layout(dataset_cfg.plates_dir, dataset_cfg.dataset_root)
    dataset_yaml = dataset_cfg.dataset_root / "fizz_dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML missing at {dataset_yaml}. Run build_dataset_remote first."
        )

    training_cfg = _training_config_for_gpu(cfg.training)

    best_model_path = run_training(
        training_cfg=training_cfg,
        dataset_cfg=dataset_cfg,
        dataset_yaml=dataset_yaml,
        plates_dir=dataset_cfg.plates_dir,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACTS_DIR / f"{training_cfg.run_name}_best.pt"
    shutil.copy2(best_model_path, destination)
    print(f"Copied best weights to {destination}")
    return destination


@app.local_entrypoint()
def main(config_path: str | None = None) -> None:
    """Convenience entrypoint to run the full pipeline sequentially with local config overrides."""
    cfg = load_pipeline_config(Path(config_path).resolve() if config_path else None)
    cfg_dict = _serialize_config(cfg)
    generate_plates_remote.call(pipeline_cfg=cfg_dict)
    build_dataset_remote.call(pipeline_cfg=cfg_dict)
    train_remote.call(pipeline_cfg=cfg_dict)

