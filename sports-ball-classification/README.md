# Sports Ball Classification - MLOps Project

An end-to-end MLOps project for classifying sports balls using Azure Machine Learning, FastAPI, Docker, and Kubernetes.

## 🏀 Project Overview

This project implements a complete MLOps pipeline for classifying 15 different types of sports balls using a Convolutional Neural Network (CNN). The system includes:

- **Azure ML Pipeline**: Automated training pipeline with data preprocessing, model training, and registration
- **FastAPI REST API**: Production-ready API for inference with database integration
- **Docker & Kubernetes**: Containerized deployment with orchestration support
- **Persistent Storage**: PostgreSQL/SQLite database for storing prediction history

## 📊 Supported Ball Categories

| Category | Category | Category |
|----------|----------|----------|
| 🏈 american_football | ⚾ baseball | 🏀 basketball |
| 🎱 billiard_ball | 🎳 bowling_ball | 🏏 cricket_ball |
| ⚽ football | ⛳ golf_ball | 🏑 hockey_ball |
| 🥅 hockey_puck | 🏉 rugby_ball | 🏸 shuttlecock |
| 🏓 table_tennis_ball | 🎾 tennis_ball | 🏐 volleyball |

## 📁 Project Structure

```
sports-ball-classification/
├── components/                 # Azure ML pipeline components
│   ├── dataprep/              # Image preprocessing & train-test split
│   │   ├── code/
│   │   │   ├── dataprep.py    # Image resizing script
│   │   │   └── traintestsplit.py
│   │   ├── conda.yaml
│   │   ├── dataprep.yaml
│   │   └── data_split.yaml
│   ├── training/              # CNN model training
│   │   ├── code/
│   │   │   ├── train.py       # Training script
│   │   │   └── utils.py       # Model architecture & utilities
│   │   ├── conda.yaml
│   │   └── training.yaml
│   └── register/              # Model registration
│       ├── code/
│       │   └── register.py
│       ├── conda.yaml
│       ├── environment.yaml
│       └── register.yaml
├── environment/               # Azure ML environment configs
│   ├── compute-cluster.yaml
│   ├── preprocessing.yaml
│   └── training.yaml
├── inference/                 # FastAPI application
│   ├── main.py               # API with database integration
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
├── kubernetes/                # K8s deployment manifests
│   └── deployment.yaml
├── pipelines/                 # Azure ML pipeline definitions
│   └── sports-ball-classification.yaml
├── setup_azure.sh            # Azure setup helper script
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Azure CLI with ML extension (`az extension add -n ml`)
- Docker & Docker Compose
- Python 3.10+
- kubectl (for Kubernetes deployment)

### 1. Azure ML Setup

```bash
# Login to Azure
az login

# Create resource group
az group create --name mlops-examen-rg --location westeurope

# Create ML workspace
az ml workspace create --name mlops-sports-ball-ws --resource-group mlops-examen-rg

# Set defaults
az configure --defaults group=mlops-examen-rg workspace=mlops-sports-ball-ws
```

### 2. Create Compute & Environments

```bash
# Create compute cluster
az ml compute create -f environment/compute-cluster.yaml

# Create environments
az ml environment create -f environment/preprocessing.yaml
az ml environment create -f environment/training.yaml
az ml environment create -f components/register/environment.yaml
```

### 3. Register Components

```bash
az ml component create -f components/dataprep/dataprep.yaml
az ml component create -f components/dataprep/data_split.yaml
az ml component create -f components/training/training.yaml
az ml component create -f components/register/register.yaml
```

### 4. Upload Dataset

```bash
# Upload each ball category
for ball in american_football baseball basketball billiard_ball bowling_ball cricket_ball football golf_ball hockey_ball hockey_puck rugby_ball shuttlecock table_tennis_ball tennis_ball volleyball; do
    az ml data create --name $ball --version 1 --path ../data/$ball --type uri_folder
done
```

### 5. Run Training Pipeline

```bash
az ml job create -f pipelines/sports-ball-classification.yaml
```

### 6. Deploy API

```bash
cd inference

# Using Docker Compose
docker-compose up -d

# Or using Docker directly
docker build -t sports-ball-api .
docker run -p 8000:8000 sports-ball-api
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API info |
| `/predict` | POST | Upload image for classification |
| `/predictions` | GET | Get prediction history |
| `/predictions/{id}` | GET | Get specific prediction |
| `/stats` | GET | Get prediction statistics |
| `/categories` | GET | List supported categories |

### Example: Classify an Image

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "img=@tennis_ball.jpg"
```

Response:
```json
{
  "id": 1,
  "predicted_label": "tennis_ball",
  "confidence": 0.97,
  "all_scores": {...},
  "created_at": "2024-01-15T10:30:00Z"
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Azure Machine Learning                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │ DataPrep │ → │  Split   │ → │ Training │ → │ Register Model   │  │
│  │ (resize) │   │(train/   │   │  (CNN)   │   │ (to Azure ML)    │  │
│  └──────────┘   │  test)   │   └──────────┘   └──────────────────┘  │
│                 └──────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                              Model Download
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Docker / Kubernetes                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                        FastAPI                                  │ │
│  │  POST /predict → CNN Inference → Store in DB → Return Result   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                │                                     │
│                                ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   PostgreSQL / SQLite                           │ │
│  │              Store prediction history & stats                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to trained model | `./model.keras` |
| `DATABASE_URL` | Database connection string | `sqlite:///./predictions.db` |
| `PORT` | API port | `8000` |

## 📝 Model Details

- **Architecture**: CNN with 3 convolutional blocks (32→64→128 filters)
- **Input Size**: 64x64 RGB images
- **Output**: 15-class softmax
- **Optimizer**: SGD with exponential decay learning rate
- **Data Augmentation**: Rotation, shifts, shear, zoom, horizontal flip

## 🧹 Cleanup

Don't forget to delete Azure resources when done to save costs:

```bash
az group delete --name mlops-examen-rg --yes --no-wait
```

## 📄 License

This project is part of an MLOps exam assignment for Howest.
