import os
import wandb
from ultralytics import YOLO

# Make sure you've done: export WANDB_API_KEY=... before running

def main():
    
    wandb.init(project="enph353-fizz-yolo", name="yolo_fizz_obstacles_v1")

    dataset_yaml = os.path.join(os.path.dirname(__file__),
                                "fizz_yolo_dataset",
                                "fizz_dataset.yaml")

    model = YOLO("yolo12s.pt")
    
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

    wandb.finish()

    # The best model weights will be something like:
    # runs/detect/enph353-fizz-yolo/yolo_fizz_obstacles_v1/weights/best.pt

if __name__ == "__main__":
    main()