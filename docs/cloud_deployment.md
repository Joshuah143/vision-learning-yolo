# Running ENPH353 Training on AWS GPU Infrastructure

This guide outlines how to move the existing Ultralytics YOLO training workflow in this repo onto managed GPU infrastructure in AWS. It compares hosting options, shows how to package the project into a reusable container, describes cloud storage/IAM wiring, and sketches infrastructure-as-code and automation patterns.

---

## 1. Pick the Right AWS Service

| Scenario | Service | Why it Fits | Considerations |
| --- | --- | --- | --- |
| Fastest path, managed training loops, easy metrics | **SageMaker Training Job** on an `ml.g5.xlarge` (A10G) or `ml.p3.2xlarge` (V100) | Handles GPU drivers, FSx/S3 mounting, checkpoints, auto-cleanup | Slightly higher per-hour cost than raw EC2, but minimal ops. Use Spot to cut 40–70%. |
| Iterative experiments, manual notebooks | **SageMaker Studio Notebook** with the same instance types | Notebook UX, built-in experiment tracking, quick W&B integration | Idle notebooks still bill. Convert to Training Jobs for long runs. |
| DIY VM, full control | **EC2** (`g5`, `p3`, `p4`) inside a VPC | You install drivers/dockers; flexible for custom libs or long-running inference | Requires managing AMIs, scaling, security patches. Use Auto Scaling Groups for fleets. |
| Batch inferencing or microservices | **ECS/EKS with GPU nodes** | Reuse the container across many tasks, integrate with queues | Needs cluster management. Better once you have steady workloads. |

**Recommendation:** Start with SageMaker Training Jobs wrapped around a custom container. It gives you managed GPUs, lifecycle logging, simple scaling, and integrates well with W&B. Fall back to single EC2 GPU instances only if you need ultra-custom drivers or want to tweak YOLO internals at the OS level.

---

## 2. Containerize the Project

1. **Standardize Python version.** Ultralytics’ GPU wheels install cleanly on Python 3.10/3.11 with CUDA 12.x. Update local testing to a matching version (e.g. `pyenv` or uv with `python = "3.10"` in `pyproject.toml`) so cloud runs match.
2. **Consolidate dependencies.** Export the lock file to a requirements list for container installs: `uv pip compile pyproject.toml -o requirements.txt` (stores pinned versions).
3. **Author a GPU-ready Dockerfile.**

   ```Dockerfile
   # syntax=docker/dockerfile:1
   FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.5.0-gpu-py310-cu121-ubuntu22.04

   # SageMaker looks for code under /opt/ml/code
   WORKDIR /opt/ml/code

   # Pre-install deps to leverage Docker layer caching
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Bring in the project
   COPY . /opt/ml/code

   # Ultralytics auto-detects CUDA. SageMaker will pass in /opt/ml/input/data.
   ENV WANDB_DISABLED=false
   ENTRYPOINT ["python", "train.py"]
   ```

4. **Build and test locally.**
   ```bash
   docker build -t enph353-yolo:latest .
   docker run --gpus all \
     -e WANDB_API_KEY=$WANDB_API_KEY \
     -v $(pwd)/fizz_yolo_dataset:/opt/ml/input/data/train \
     enph353-yolo:latest
   ```
5. **Publish to Amazon ECR.** Create a repo (e.g. `enph353-yolo-training`), authenticate with `aws ecr get-login-password`, then push `aws ecr create-repository --repository-name enph353-yolo-training` and `docker push`.

---

## 3. Stage Data and Manage IAM

1. **Organize S3.** Create a bucket like `s3://enph353-ml-artifacts` with prefixes:
   ```
   datasets/fizz_yolo_dataset/...
   models/pretrained/yolo12s.pt
   outputs/<job-name>/weights/
   ```
   Sync the current dataset:  
   `aws s3 sync fizz_yolo_dataset s3://enph353-ml-artifacts/datasets/fizz_yolo_dataset`
2. **Parameterize data paths.** Allow `train.py` to read from env vars:
   ```python
10:32:train.py
dataset_root = Path(os.getenv("DATASET_ROOT", Path(__file__).parent / "fizz_yolo_dataset"))
dataset_yaml = dataset_root / "fizz_dataset.yaml"
weights_uri = os.getenv("BASE_WEIGHTS", "yolo12s.pt")
model = YOLO(weights_uri)
```
   When SageMaker mounts channels, the YAML can live at `/opt/ml/input/data/train/fizz_dataset.yaml`.
3. **IAM roles.** Create a SageMaker execution role with:
   - `AmazonS3ReadOnlyAccess` for the dataset prefix.
   - Scoped write access to `s3://enph353-ml-artifacts/outputs/*`.
   - Permission to call `logs:*` and `cloudwatch:*` for monitoring.
   Lock down with least privilege by restricting the ARN to the bucket prefix only.
4. **Secrets.** Store W&B keys in AWS Secrets Manager, surface as env vars:
   - Attach `secretsmanager:GetSecretValue` to the role.
   - In the job definition, add `Environment={ "WANDB_API_KEY": "{{resolve:secretsmanager:enph353/wandb}}" }`.

---

## 4. Automate Launch & Observability

### SageMaker Training Job Template (CLI)
```bash
aws sagemaker create-training-job \
  --training-job-name enph353-yolo-$(date +%Y%m%d%H%M) \
  --role-arn arn:aws:iam::<acct-id>:role/SageMakerExecutionRole \
  --algorithm-specification TrainingImage=<account>.dkr.ecr.<region>.amazonaws.com/enph353-yolo-training:latest,TrainingInputMode=File \
  --input-data-config '[{"ChannelName":"train","DataSource":{"S3DataSource":{"S3Uri":"s3://enph353-ml-artifacts/datasets/fizz_yolo_dataset","S3DataType":"S3Prefix","S3DataDistributionType":"FullyReplicated"}}}]' \
  --output-data-config S3OutputPath=s3://enph353-ml-artifacts/outputs/ \
  --resource-config InstanceType=ml.g5.2xlarge,InstanceCount=1,VolumeSizeInGB=100 \
  --stopping-condition MaxRuntimeInSeconds=14400 \
  --enable-managed-spot-training \
  --environment '{ "WANDB_API_KEY": "{{resolve:secretsmanager:enph353/wandb}}", "DATASET_ROOT": "/opt/ml/input/data/train", "BASE_WEIGHTS": "s3://enph353-ml-artifacts/models/pretrained/yolo12s.pt" }'
```

### Terraform/IaC Pointers
* Use the `aws_sagemaker_training_job` resource with variables for instance type, image URI, and S3 paths.
* Add `aws_s3_bucket` + versioning + lifecycle rules (move checkpoints to Glacier after 30 days).
* Wire CloudWatch alarms on `MaxRuntimeInSeconds` and `SpotInterruption` notifications (route to Slack/Email via SNS).
* Optional: create an EventBridge rule that triggers a job when pushing to a `main` branch tag; the payload can include hyperparameters via the `HyperParameters` field.

### CI/CD Integration
1. **Build & push container** on every main-branch merge via GitHub Actions using OIDC to assume an IAM role without long-term keys.
2. **Automated training triggers** (manual approval step recommended) to prevent runaway costs.
3. **Artifacts**: fetch `best.pt` from `S3://.../outputs/<job>/` after job success and promote to inference.

### Monitoring & Cost Guardrails
* Enable SageMaker Debugger/Profiler for GPU utilization insights.
* Pipe metrics to W&B (already integrated) and CloudWatch. Create dashboards for loss curves and GPU usage.
* Use Service Quotas and AWS Budgets alarms (daily cost ceiling for GPU instances).

---

## 5. Is Cloud Hosting Worth It?

**Pros**
* Access to modern NVIDIA GPUs without buying hardware.
* Managed spot instances can drop costs substantially for batch training.
* Infrastructure-as-code + containers → reproducible experiments and collaboration.

**Trade-offs**
* Needs upfront time to containerize and script jobs.
* On-demand GPU hourly rates add up if you leave instances running (use auto-stopping notebooks or training jobs).
* Data egress charges if you frequently pull large checkpoints locally.

For intermittent model training or when you need more GPU than a laptop offers, SageMaker Training Jobs strike the best balance of flexibility and operational simplicity. If you plan to serve the YOLO model continuously (e.g., real-time inference), extend the same container to SageMaker Endpoints or ECS Fargate with GPU to keep the environment consistent.

---

## Next Steps Checklist
1. Update the code to support `DATASET_ROOT`/`BASE_WEIGHTS` env vars and re-run locally.
2. Generate `requirements.txt`, build the Docker image, and push to ECR.
3. Create S3 bucket + IAM role. Store W&B keys in Secrets Manager.
4. Launch a SageMaker Training Job (spot + g5.xlarge recommended) and confirm outputs sync to S3.
5. Add Terraform/CloudFormation templates + CI workflows when comfortable with manual runs.

Once these steps are in place, running training at scale becomes a single CLI call or GitHub workflow, and you can iterate on hyperparameters while keeping experiments reproducible and cost-controlled.

