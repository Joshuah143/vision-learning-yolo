# ENPH353 Modal Pipeline

This repository trains the fizz sign detector entirely on [Modal](https://modal.com). Every stage—plate rendering, dataset generation, and YOLO training—is driven by `pipeline_config.yaml`, and all data persists inside the Modal volume `enph353-training-data`.

## 1. One-Time Setup

1. **Install tooling (locally):**
   ```bash
   pip install uv modal
   modal setup
   ```
2. **Configure Modal storage:**
   ```bash
   uv run modal volume create enph353-training-data
   ```
3. **(Recommended) Add your Weights & Biases key:**
   ```bash
   uv run modal secret create wandb-api-key --env WANDB_API_KEY=xxxx
   ```
   `train_remote` automatically consumes this secret.

## 2. Configure the Pipeline

Edit `pipeline_config.yaml`. This file controls:

- Plate count, seed, and output directory (`plates` block)
- Dataset paths, augmentation counts, image size, and class names (`dataset` block)
- Training hyperparameters and run naming (`training` block)

Relative paths are resolved from the repo root. The Modal pipeline mirrors those paths inside the shared volume automatically; you **do not** need to edit anything else for cloud runs.

## 3. Run on Modal

All commands below must be executed from the repo root (`/Users/joshuahimmens/code/enph353_training`).

### 3.1 Full end-to-end run (plates → dataset → train)

```bash
uv run modal run modal_pipeline.py
```

The local entrypoint uploads your **local** `pipeline_config.yaml` (or a custom file via `--config-path`) to each remote call instead of mounting the filesystem.

### 3.2 Running individual stages

```bash
# Plates only
uv run modal run modal_pipeline.py::generate_plates_remote

# Dataset only
uv run modal run modal_pipeline.py::build_dataset_remote

# Train only
uv run modal run modal_pipeline.py::train_remote
```

When invoked directly, each function reads the baked-in `pipeline_config.yaml`. When orchestrated through the entrypoint, they receive the live config payload from your machine.

## 4. Retrieve Artifacts

The trainer copies `best.pt` into `/vol/enph353/artifacts`. Download it locally with:

```bash
uv run modal volume cp enph353-training-data:/vol/enph353/artifacts/<run_name>_best.pt ./best.pt
```

Replace `<run_name>` with the `training.run_name` from `pipeline_config.yaml`.

## 5. Troubleshooting

- **Missing dataset YAML?** Ensure you have run `modal run modal_pipeline.py::build_dataset_remote` after generating plates. The training step expects `fizz_dataset.yaml` inside the dataset root specified in the config.
- **WANDB authentication errors?** Confirm you created the Modal secret named `wandb-api-key`. Alternatively, disable WANDB by editing `train.py`.
- **Changed config not reflected?** Modal caches images; if you added new Python dependencies, re-run `modal run ...` and Modal will rebuild automatically. For purely YAML changes, simply rerun the desired function.

That’s it—you now have a repeatable, config-driven Modal pipeline for generating synthetic fizz-sign data and training YOLO on an H100.

