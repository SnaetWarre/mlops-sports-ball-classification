# MLOps Exam Assignment - Progress Tracker

## Assignment Overview

Build an MLOps project with:
- A Kaggle dataset (or own project)
- Azure Machine Learning Service for cloud training
- FastAPI deployment on Docker and Kubernetes
- Optional: Database for persistent storage ✅ (We're doing this!)

## Deadline
**2 days before the lab and theory exam**

---

## Project Details

### Dataset: Sports Ball Classification 🏀⚽🎾

A multi-class image classification dataset with **15 categories** of sports balls:

| Ball Type | # Images |
|-----------|----------|
| american_football | 96 |
| baseball | 100 |
| basketball | 86 |
| billiard_ball | 162 |
| bowling_ball | 111 |
| cricket_ball | 146 |
| football (soccer) | 151 |
| golf_ball | 138 |
| hockey_ball | 133 |
| hockey_puck | 98 |
| rugby_ball | 124 |
| shuttlecock | 108 |
| table_tennis_ball | 156 |
| tennis_ball | 123 |
| volleyball | 109 |

**Total: ~1,841 images**

### Model Type
**CNN (Convolutional Neural Network)** - TensorFlow/Keras
- Easy, fast, scales well on CPU infrastructure
- Same architecture as the animal classification project
- Adapted for 15 output classes instead of 3

### Company Integration Story
**TBD** - Possible ideas:
- Sports equipment retail company wanting to auto-categorize product images
- Sports analytics company for game footage analysis
- Sports inventory management system
- Mobile app for sports enthusiasts to identify equipment

---

## Progress Checklist

### Phase 1: Project Setup
- [x] Choose a Kaggle dataset / project ✅ Sports Ball Classification
- [x] Define the AI model type ✅ CNN (TensorFlow/Keras)
- [x] Create project structure ✅ `sports-ball-classification/` folder
- [x] Create GitHub Actions workflows ✅ Full CI/CD automation!
- [ ] Create fictional/real company integration story
- [ ] Clean up old Azure resources (from previous animal classification project)

### Phase 2: Azure Machine Learning (via GitHub Actions)
- [ ] Set up `AZURE_CREDENTIALS` secret in GitHub
- [ ] Run `azure-ml-pipeline.yml` workflow with `full-pipeline` action
- [ ] Verify Resource Group created
- [ ] Verify ML Workspace created
- [ ] Verify Compute Cluster created
- [ ] Verify Environments created
- [ ] Verify Components registered
- [ ] Verify Datasets uploaded (15 ball categories)
- [ ] Verify Training Pipeline runs
- [ ] Verify Model is registered in Azure ML

### Phase 3: FastAPI & Docker
- [x] Create FastAPI application for inference ✅
- [x] Add database integration (SQLAlchemy + PostgreSQL) ✅
- [x] Create Dockerfile ✅
- [x] Create docker-compose.yml (with PostgreSQL) ✅
- [ ] Download trained model from Azure ML
- [ ] Test locally with Docker

### Phase 4: Kubernetes Deployment
- [x] Create Kubernetes deployment YAML ✅
- [x] Create Kubernetes service YAML ✅
- [ ] Deploy to Kubernetes cluster
- [ ] Test endpoint

### Phase 5: Database Integration (FOR MAXIMUM POINTS! 🎯)
- [x] Choose database (PostgreSQL for production, SQLite for dev) ✅
- [x] Design schema for storing prediction results ✅
- [x] Integrate database with FastAPI ✅
- [x] Add endpoints for retrieving prediction history ✅
- [x] Include in Docker Compose setup ✅

### Phase 6: CI/CD Automation (GitHub Actions)
- [x] Create `azure-ml-pipeline.yml` workflow ✅
  - [x] Setup infrastructure job
  - [x] Create environments job
  - [x] Register components job
  - [x] Upload datasets job
  - [x] Run training job
  - [x] Cleanup job
- [x] Create `deploy-inference.yml` workflow ✅
  - [x] Deploy action
  - [x] Stop action
  - [x] Restart action
  - [x] Logs action
  - [x] Test action
- [ ] Set up self-hosted runner on local machine
- [ ] Test full CI/CD pipeline

---

## GitHub Actions Workflows

### 1. `azure-ml-pipeline.yml` - Azure ML Pipeline
**Runs on:** `ubuntu-latest` (GitHub-hosted)

| Action | Description |
|--------|-------------|
| `full-pipeline` | Setup Azure + Upload data + Run training |
| `setup-only` | Only create Azure resources |
| `train-only` | Only run training pipeline |
| `cleanup` | Delete all Azure resources |

**Triggers:**
- Manual dispatch
- Push to `main`/`master` (changes to `components/`, `pipelines/`, `environment/`, `data/`)

### 2. `deploy-inference.yml` - Local Inference Deployment
**Runs on:** `self-hosted` (your local machine)

| Action | Description |
|--------|-------------|
| `deploy` | Build and start Docker containers |
| `stop` | Stop all containers |
| `restart` | Stop and restart containers |
| `logs` | Show container logs |
| `test` | Run API tests |

**Triggers:**
- Manual dispatch
- Push to `main`/`master` (changes to `inference/`)

---

## Report Checklist

- [ ] Small explanation of the project and data
- [ ] Screenshots of Cloud AI services OR short demo video
- [ ] FastAPI explanation and integration possibilities
- [ ] Extra's of Kubernetes (if applicable)
- [ ] Extra's around automation (GitHub Actions CI/CD!)

---

## Final Submission Checklist

- [ ] Report completed
- [ ] Source code zipped
- [ ] Azure resources cleaned up (to save costs!)
- [ ] Submitted before deadline

---

## Setup Instructions

### 1. Set up Azure Credentials Secret

```bash
# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-sports-ball" \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID> \
  --sdk-auth
```

Add the JSON output as `AZURE_CREDENTIALS` secret in GitHub:
**Settings → Secrets and variables → Actions → New repository secret**

### 2. Set up Self-Hosted Runner

1. Go to **Settings → Actions → Runners → New self-hosted runner**
2. Follow instructions to download and configure
3. Start the runner: `./run.sh` or install as service

### 3. Run the Pipeline

1. Go to **Actions** tab
2. Select **Azure ML Pipeline**
3. Click **Run workflow**
4. Select `full-pipeline` action
5. Wait for completion

### 4. Deploy Inference API

1. Download model from Azure ML
2. Place in `inference/model/sports-ball-cnn/model.keras`
3. Run **Deploy Inference API** workflow with `deploy` action

---

## Notes & Decisions

- Deleted `__MACOSX` folder from dataset (macOS metadata, not needed)
- Using same CNN architecture as animal classification (proven to work)
- Adapted for 15 output classes instead of 3
- Including PostgreSQL database for maximum points!
- API includes: predictions, history, statistics, health endpoints
- **Full CI/CD through GitHub Actions - no manual scripts needed!**

---

## Current Status

**Status**: 🟡 In Progress - Workflows created, ready for Azure setup

**Next Steps**:
1. ⏳ Delete old Azure workspace (user doing this)
2. Set up `AZURE_CREDENTIALS` secret in GitHub
3. Run `azure-ml-pipeline.yml` workflow
4. Set up self-hosted runner for inference
5. Download trained model
6. Deploy inference API
7. Write report