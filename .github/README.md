# 🏀 MLOps Sports Ball Classification - GitHub Actions

This repository contains a unified MLOps pipeline that automates the entire machine learning lifecycle for sports ball image classification.

## 🔄 Unified Workflow

The single workflow `mlops-pipeline.yml` handles everything:

### Pipeline Stages

1. **🏗️ Setup Azure Infrastructure**
   - Creates/verifies Resource Group
   - Creates/verifies Azure ML Workspace
   - Creates/verifies Compute Cluster
   - Registers ML Environments (preprocessing, training, register)
   - Registers ML Components (dataprep, data_split, training, register_model)
   - Uploads training datasets (15 ball categories)

2. **🚀 Train Model on Azure ML**
   - Submits the training pipeline to Azure ML
   - Waits for completion (optional)
   - Outputs job name for tracking

3. **📥 Download Trained Model**
   - Runs on self-hosted runner (your local machine)
   - Downloads the latest model version from Azure ML
   - Places model in `sports-ball-classification/inference/model/`

4. **🐳 Deploy Inference API**
   - Runs on self-hosted runner
   - Builds and starts Docker containers
   - Deploys FastAPI inference service with PostgreSQL
   - Runs health checks and tests

## 🎯 Available Actions

Trigger the workflow manually with these actions:

| Action | Description |
|--------|-------------|
| `full-pipeline` | Run everything: setup → train → download → deploy |
| `train-only` | Setup Azure and run training (no local deployment) |
| `download-model` | Download latest model from Azure ML to local machine |
| `deploy-inference` | Deploy the inference API locally |
| `cleanup` | Delete all Azure resources |

## ⚙️ Configuration

### Workflow Inputs

- **epochs**: Number of training epochs (default: 10)
- **wait_for_completion**: Wait for training before downloading model (default: true)

### Required Secrets

- `AZURE_CREDENTIALS`: Azure Service Principal credentials (JSON format)

### Environment Variables

```yaml
RESOURCE_GROUP: mlops-sports-ball-rg
WORKSPACE_NAME: mlops-sports-ball-ws
LOCATION: westeurope
COMPUTE_CLUSTER_NAME: sports-ball-cluster
MODEL_NAME: sports-ball-cnn
```

## 🖥️ Self-Hosted Runner

The download and deployment stages require a self-hosted GitHub Actions runner on your local machine.

### Setup Runner

```bash
# Download and configure runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64.tar.gz

# Configure (get token from GitHub repo settings)
./config.sh --url https://github.com/YOUR_USER/YOUR_REPO --token YOUR_TOKEN

# Start runner
./run.sh
```

### Requirements on Runner Machine

- Docker & Docker Compose
- Azure CLI (`az`)
- Python 3.x
- curl

## 📡 API Endpoints

After deployment, the API is available at:

- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Predict**: http://localhost:8000/predict (POST with image)
- **Categories**: http://localhost:8000/categories
- **Statistics**: http://localhost:8000/stats

## 🚀 Quick Start

1. Ensure Azure credentials are set in repository secrets
2. Start your self-hosted runner locally
3. Trigger the workflow:
   ```bash
   gh workflow run mlops-pipeline.yml --field action=full-pipeline --field epochs=10
   ```
4. Monitor progress in GitHub Actions tab
5. Access API at http://localhost:8000/docs when complete

## 📁 Project Structure

```
├── .github/workflows/
│   └── mlops-pipeline.yml     # Unified MLOps workflow
├── data/                       # Training images (15 ball categories)
├── sports-ball-classification/
│   ├── components/            # Azure ML components
│   ├── environment/           # ML environment configs
│   ├── inference/             # FastAPI + Docker
│   ├── kubernetes/            # K8s deployment manifests
│   └── pipelines/             # Azure ML pipeline definition
└── assignment/                 # Reference material
```
