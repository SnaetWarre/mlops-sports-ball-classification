# 🏀 MLOps Project Report: Sports Ball Classification

**Student:** Warre Snaet
**Course:** MLOps
**Date:** December 1, 2025

---

## 1. Project Explanation

### The Concept
This project focuses on the automated classification of sports balls using Computer Vision. The goal is to accurately identify 15 different types of balls (e.g., basketball, tennis ball, rugby ball) from images.

### Integration Story (Fictional Company)
**"SportScan Analytics"** is a fictional company that provides automated inventory management solutions for large sports retailers and gymnasiums.
- **Problem**: Manually counting and categorizing thousands of mixed sports items in a warehouse is time-consuming and error-prone.
- **Solution**: An automated camera system on a conveyor belt takes pictures of items. Our AI model identifies the type of ball, and the system automatically updates the inventory database.
- **Integration**: The FastAPI endpoint serves as the brain of this system. The warehouse software sends an image to the API, receives the classification, and logs the item into the PostgreSQL database for real-time stock tracking.

### The Data
The dataset is sourced from Kaggle and consists of approximately **1,841 images** across **15 categories**.
- **Preprocessing**: Images are resized to a standard resolution (e.g., 128x128) and normalized (pixel values scaled to 0-1) to ensure consistent input for the neural network.
- **Data Augmentation**: To improve model robustness, we apply random transformations (rotation, zoom, flip) during training.

### The AI Model
We implemented a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras**.
- **Architecture**: A sequential model with multiple Conv2D and MaxPooling2D layers to extract features (shapes, textures), followed by Dense layers for classification.
- **Output**: A Softmax layer with 15 neurons, providing a probability distribution across the 15 ball types.

---

## 2. Cloud AI Services (Azure Machine Learning)

We utilized **Azure Machine Learning Service** to move from local experimentation to a professional, scalable training pipeline.

### Key Components Used
1.  **Compute Clusters**: We used scalable `Standard_DS11_v2` clusters. This allows us to train on powerful hardware without maintaining it. The cluster scales down to 0 nodes when not in use to save costs.
2.  **Environments**: We defined custom Docker environments (`environment/training.yaml`) to ensure that our training code runs with the exact same dependencies (TensorFlow, NumPy, etc.) every time, regardless of where it's executed.
3.  **Data Assets**: The dataset was uploaded and registered as a versioned Data Asset in Azure ML. This ensures data reproducibility—we always know exactly which version of the data was used to train a specific model.
4.  **Pipelines**: The entire training process (Data Prep -> Train -> Register) is defined as a YAML pipeline (`pipelines/sports-ball-classification.yaml`), making it reproducible and automated.

*(Please insert screenshots of your Azure ML Studio here: Pipeline Run, Registered Model, and Compute Cluster)*

---

## 3. FastAPI & Integration

To make our model usable by "SportScan Analytics," we wrapped it in a **FastAPI** application.

### API Design
- **`/predict` (POST)**: The core endpoint. It accepts an image file upload.
    - **Process**: The API receives the image -> Preprocesses it (resize/normalize) -> Feeds it to the loaded Keras model -> Returns the predicted class and confidence score.
- **`/history` (GET)**: Returns a list of past predictions stored in the database.
- **`/health` (GET)**: A simple health check for container orchestration systems (like Kubernetes) to verify the service is up.

### Database Integration (Extra Feature)
We went a step further by integrating a **PostgreSQL** database using SQLAlchemy.
- Every time a prediction is made, the result (filename, predicted class, confidence, timestamp) is automatically saved to the database.
- This allows "SportScan Analytics" to generate reports on which items are most frequently processed.

### Integration with Other Services
This API is containerized with Docker, meaning it can be deployed anywhere (Cloud, On-Premise, Edge Device).
- **Warehouse System**: Can call the API via HTTP requests.
- **Mobile App**: Staff can use a mobile app to snap a photo of a stray ball, and the API will tell them which bin it belongs to.

---

## 4. Kubernetes

While our primary deployment for this assignment uses Docker Compose on a self-hosted runner, the project is designed to be **Cloud Native**.

- **Deployment Manifests**: We created `kubernetes/deployment.yaml` and `kubernetes/service.yaml`.
- **Scalability**: In a production scenario (e.g., Azure Kubernetes Service), we could easily scale the number of API replicas based on CPU usage to handle high traffic from multiple warehouses simultaneously.
- **Service Discovery**: The Kubernetes Service ensures that internal traffic is load-balanced across all healthy API pods.

---

## 5. Automation (GitHub Actions)

We implemented a fully automated **CI/CD (Continuous Integration / Continuous Deployment)** pipeline to ensure reliability and speed.

### The Pipeline (`.github/workflows/mlops-pipeline.yml`)
Our pipeline is triggered automatically on every push to the `master` branch. It consists of distinct stages:

1.  **🔐 Configure Permissions (Self-Hosted)**:
    - **Innovation**: We solved the "Chicken and Egg" permission problem by running a setup script on our self-hosted runner first. This script uses our local Azure credentials to grant the necessary "Owner" rights to the Service Principal, allowing the subsequent cloud steps to assign roles to compute clusters without manual intervention.

2.  **🏗️ Infrastructure as Code (Cloud)**:
    - The pipeline automatically provisions or updates the Azure resources (Resource Group, Workspace, Compute) using the Azure CLI.
    - It registers the ML environments and components.

3.  **🚀 Continuous Training (Cloud)**:
    - It submits the training job to Azure ML. If the data or code has changed, a new model is trained automatically.

4.  **📥 Model Delivery (Hybrid)**:
    - Once training is complete, the pipeline downloads the new model artifact from the cloud to our local machine (Self-Hosted Runner).

5.  **🐳 Continuous Deployment (Local)**:
    - The pipeline builds the new Docker images (API + Database).
    - It uses `docker-compose` to restart the services with the new model version.
    - It runs a health check to verify the deployment was successful.

This level of automation means that a data scientist can simply push code changes to GitHub, and within minutes, a new model is trained in the cloud and deployed to the production edge device without any manual commands.

---

## 6. Source Code

The full source code is attached in the accompanying ZIP file, structured as follows:
- `components/`: Azure ML component definitions (Data Prep, Training).
- `data/`: The sports ball dataset.
- `environment/`: Conda and Docker environment specifications.
- `inference/`: FastAPI application, Dockerfile, and Database logic.
- `kubernetes/`: K8s deployment manifests.
- `pipelines/`: Azure ML pipeline definitions.
- `scripts/`: Helper scripts for setup and permissions.
- `.github/workflows/`: CI/CD pipeline configurations.
