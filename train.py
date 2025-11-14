#!/usr/bin/env python3
import os
import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

# Make sure you've done: export WANDB_API_KEY=... before running

def main():
    # Initialize wandb
    wandb.init(project="enph353-fizz-yolo", name="yolo_fizz_obstacles_v1")

    # Path to dataset yaml generated in script 2
    dataset_yaml = os.path.join(os.path.dirname(__file__),
                                "fizz_yolo_dataset",
                                "fizz_dataset.yaml")

    model = YOLO("yolo12s.pt")
    # add_wandb_callback(model, enable_model_checkpointing=True)

    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=50,              # adjust as needed
        imgsz=640,
        batch=16,
        project="enph353-fizz-yolo",
        name="yolo_fizz_obstacles_v1",
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        patience=20,            # early stopping if val plateaus
        verbose=True,
        device='mps',               # GPU index
        exist_ok=True,
        # W&B integration
        val=True
    )

    # The ultralytics + wandb integration automatically logs:
    # metrics, losses, some images, etc.

    # Finish wandb run
    wandb.finish()

    # The best model weights will be something like:
    # runs/detect/enph353-fizz-yolo/yolo_fizz_obstacles_v1/weights/best.pt

if __name__ == "__main__":
    main()