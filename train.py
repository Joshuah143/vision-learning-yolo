#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import wandb
from ultralytics import YOLO, settings

from gen_dataset import DatasetConfig, collect_plate_samples
from pipeline_config import DatasetSectionConfig, TrainingConfig, load_pipeline_config


def run_training(
    training_cfg: TrainingConfig | None = None,
    dataset_cfg: DatasetSectionConfig | None = None,
    *,
    dataset_yaml: Path | None = None,
    plates_dir: Path | None = None,
) -> Path:
    """Train YOLO using configuration sourced from pipeline_config.yaml."""
    if training_cfg is None or dataset_cfg is None:
        pipeline_cfg = load_pipeline_config()
        if training_cfg is None:
            training_cfg = pipeline_cfg.training
        if dataset_cfg is None:
            dataset_cfg = pipeline_cfg.dataset

    script_dir = Path(__file__).resolve().parent
    dataset_yaml = (dataset_yaml or (dataset_cfg.dataset_root / "fizz_dataset.yaml")).resolve()
    dataset_root = dataset_cfg.dataset_root
    plates_dir = (plates_dir or dataset_cfg.plates_dir).resolve()

    # Ensure Ultralytics has WANDB logging enabled (equivalent to `yolo settings wandb=True`)
    yolo_settings = settings
    if not bool(yolo_settings.get("wandb", False)):
        yolo_settings["wandb"] = True
        yolo_settings.save()

    run = wandb.init(project=training_cfg.project, name=training_cfg.run_name)

    model = YOLO(str(training_cfg.weights_path))

    train_args: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": training_cfg.epochs,
        "imgsz": training_cfg.imgsz,
        "batch": training_cfg.batch,
        "project": training_cfg.project,
        "name": training_cfg.run_name,
        "pretrained": True,
        "optimizer": training_cfg.optimizer,
        "lr0": training_cfg.lr0,
        "patience": training_cfg.patience,
        "verbose": True,
        "device": training_cfg.device,
        "exist_ok": True,
        "val": True,
    }

    results = model.train(**train_args)

    weights_dir = Path(results.save_dir) / "weights"
    best_model_path = weights_dir / "best.pt"

    dataset_config = DatasetConfig(dataset_root=dataset_root, plates_dir=plates_dir)
    plates = (
        collect_plate_samples(dataset_config.plates_dir) if dataset_config.plates_dir.exists() else []
    )

    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=script_dir)
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    artifact_metadata = {
        "epochs": train_args["epochs"],
        "imgsz": train_args["imgsz"],
        "batch": train_args["batch"],
        "optimizer": train_args["optimizer"],
        "lr0": train_args["lr0"],
        "patience": train_args["patience"],
        "dataset_root": str(dataset_config.dataset_root),
        "num_plate_templates": len(plates),
        "num_class_names": len(dataset_config.class_names),
        "letter_classes": [name for name in dataset_config.class_names if name.startswith("letter_")],
        "git_hash": git_hash,
    }

    artifact = wandb.Artifact(
        name="yolo-sign-letters",
        type="model",
        description="YOLO model trained on fizz sign dataset with per-letter labels.",
        metadata=artifact_metadata,
    )

    if best_model_path.exists():
        artifact.add_file(str(best_model_path))
    else:
        wandb.termwarn(f"Best model weights not found at {best_model_path}")

    run.log_artifact(artifact)
    wandb.finish()

    return best_model_path


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
