# GitHub Actions CI/CD Setup Guide

This guide explains how to set up the GitHub Actions workflows for the Sports Ball Classification MLOps project.

## 🔧 Prerequisites

1. An Azure account with an active subscription
2. A GitHub repository with this code
3. Docker installed on your local machine (for self-hosted runner)

## 📁 Workflows

### 1. `azure-ml-pipeline.yml` - Azure ML Pipeline
Runs on GitHub-hosted runners (`ubuntu-latest`) and handles:
- Creating Azure Resource Group
- Creating Azure ML Workspace
- Creating Compute Cluster
- Creating Environments
- Registering Components
- Uploading Datasets
- Running Training Pipeline
- Cleanup (optional)

**Triggers:**
- Manual dispatch (`workflow_dispatch`)
- Push to `main`/`master` on changes to `components/`, `pipelines/`, `environment/`, or `data/`

**Actions:**
| Action | Description |
|--------|-------------|
| `full-pipeline` | Run everything from setup to training |
| `setup-only` | Only setup Azure resources (no training) |
| `train-only` | Only run training (assumes setup is done) |
| `cleanup` | Delete all Azure resources |

### 2. `deploy-inference.yml` - Deploy Inference API
Runs on **self-hosted runner** (your local machine) and handles:
- Building Docker images
- Starting/stopping containers
- Health checks
- API testing

**Triggers:**
- Manual dispatch (`workflow_dispatch`)
- Push to `main`/`master` on changes to `inference/`

**Actions:**
| Action | Description |
|--------|-------------|
| `deploy` | Build and start the API |
| `stop` | Stop the API containers |
| `restart` | Stop and restart the API |
| `logs` | Show container logs |
| `test` | Run API tests |

## 🔐 Setting Up Azure Credentials

### Step 1: Create a Service Principal

```bash
# Login to Azure
az login

# Create service principal with contributor access
az ad sp create-for-rbac \
  --name "github-actions-sports-ball" \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID> \
  --sdk-auth
```

This will output JSON like:
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  ...
}
```

### Step 2: Add Secret to GitHub

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `AZURE_CREDENTIALS`
5. Value: Paste the entire JSON output from Step 1
6. Click **Add secret**

## 🏃 Setting Up Self-Hosted Runner

The inference deployment runs on your local machine using a self-hosted GitHub Actions runner.

### Step 1: Download and Install Runner

```bash
# Create a directory for the runner
mkdir ~/actions-runner && cd ~/actions-runner

# Download the latest runner (check GitHub for latest version)
curl -o actions-runner-linux-x64-2.321.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz

# Extract
tar xzf actions-runner-linux-x64-2.321.0.tar.gz
```

### Step 2: Configure the Runner

1. Go to your GitHub repository
2. Navigate to **Settings** → **Actions** → **Runners**
3. Click **New self-hosted runner**
4. Select your OS (Linux/macOS/Windows)
5. Copy the configuration command (includes your token)
6. Run it in the `~/actions-runner` directory

```bash
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN
```

### Step 3: Start the Runner

```bash
# Run interactively (stops when terminal closes)
./run.sh

# OR install as a service (recommended)
sudo ./svc.sh install
sudo ./svc.sh start
```

### Step 4: Verify Runner is Online

1. Go to **Settings** → **Actions** → **Runners**
2. Your runner should show as "Idle" (green)

## 🚀 Running the Workflows

### Option 1: Via GitHub UI

1. Go to **Actions** tab in your repository
2. Select the workflow you want to run
3. Click **Run workflow**
4. Select the action and parameters
5. Click **Run workflow**

### Option 2: Via Push

Simply push changes to the relevant directories:
- Push to `components/`, `pipelines/`, `environment/`, or `data/` → Triggers Azure ML Pipeline
- Push to `inference/` → Triggers Inference Deployment

## 📋 Workflow Execution Order

For a complete MLOps pipeline:

1. **First Time Setup:**
   ```
   Azure ML Pipeline (full-pipeline)
   └── Creates all Azure resources
   └── Uploads datasets
   └── Runs training
   └── Registers model
   ```

2. **After Training Completes:**
   - Download the model from Azure ML
   - Place it in `inference/model/sports-ball-cnn/model.keras`

3. **Deploy Inference:**
   ```
   Deploy Inference (deploy)
   └── Builds Docker images
   └── Starts containers
   └── Runs health check
   ```

## 🔍 Monitoring

### Azure ML Jobs
```bash
# List recent jobs
az ml job list --output table

# Show specific job
az ml job show --name <job-name>

# Stream job logs
az ml job stream --name <job-name>
```

### Local Inference
```bash
# Check container status
cd inference && docker-compose ps

# View logs
docker-compose logs -f

# Test API
curl http://localhost:8000/health
```

## ⚠️ Troubleshooting

### Azure Login Fails
- Verify `AZURE_CREDENTIALS` secret is set correctly
- Check if the service principal has expired
- Ensure the service principal has contributor access

### Self-Hosted Runner Not Picking Up Jobs
- Check runner status in GitHub Settings
- Verify runner is running: `sudo ./svc.sh status`
- Check runner logs: `cat _diag/*.log`

### Docker Compose Fails
- Ensure Docker is running: `docker info`
- Check if ports are already in use: `lsof -i :8000`
- View container logs: `docker-compose logs`

## 🧹 Cleanup

### Stop Inference API
Run the `deploy-inference.yml` workflow with action `stop`, or manually:
```bash
cd inference && docker-compose down
```

### Delete Azure Resources
Run the `azure-ml-pipeline.yml` workflow with action `cleanup`, or manually:
```bash
az group delete --name mlops-sports-ball-rg --yes
```

### Remove Self-Hosted Runner
```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token <YOUR_TOKEN>
```
