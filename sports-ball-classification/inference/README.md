# Sports Ball Classification API

A FastAPI-based REST API for classifying sports balls using a trained CNN model. This API supports 15 different types of sports balls and includes database integration for storing prediction history.

## Features

- 🏀 **15 Ball Categories**: american_football, baseball, basketball, billiard_ball, bowling_ball, cricket_ball, football, golf_ball, hockey_ball, hockey_puck, rugby_ball, shuttlecock, table_tennis_ball, tennis_ball, volleyball
- 📊 **Prediction History**: All predictions are stored in a database for analytics and history tracking
- 🔥 **Grad-CAM Dashboard**: Interactive Streamlit dashboard with explainability visualizations
- 🐳 **Docker Ready**: Includes Dockerfile and docker-compose for easy deployment
- ☸️ **Kubernetes Ready**: Deployment manifests included
- 📈 **Statistics Endpoint**: Get insights about prediction patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose                               │
│                                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │    FastAPI      │  │    Streamlit    │  │     PostgreSQL      │  │
│  │   (port 8000)   │  │   (port 8501)   │  │     (port 5432)     │  │
│  │                 │  │                 │  │                     │  │
│  │  REST API for   │  │  Grad-CAM       │  │  Prediction         │  │
│  │  predictions    │  │  Dashboard      │  │  History            │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
│           │                    │                       │             │
│           └────────────────────┴───────────────────────┘             │
│                            model.keras                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A trained model file (`model.keras`)

### Running with Docker Compose

1. Place your trained model in the `model/sports-ball-classification/` directory:
   ```
   inference/
   ├── model/
   │   └── sports-ball-classification/
   │       └── model.keras
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Access the services:
   - **REST API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs
   - **Grad-CAM Dashboard**: http://localhost:8501

### Running Locally (Development)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export MODEL_PATH=/path/to/model.keras
   export DATABASE_URL=sqlite:///./predictions.db
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## 🔥 Grad-CAM Dashboard

The interactive dashboard provides **explainability** for model predictions using Grad-CAM (Gradient-weighted Class Activation Mapping).

### Features

- **Image Upload**: Drag & drop sports ball images
- **Real-time Classification**: Instant predictions with confidence scores
- **Heatmap Visualization**: See exactly what the model is "looking at"
- **Adjustable Settings**: Control heatmap intensity and colormap

### Running the Dashboard

```bash
# With Docker Compose (starts all services)
docker-compose up -d

# Or standalone
cd dashboard
streamlit run app.py
```

Then open http://localhost:8501

### What is Grad-CAM?

Grad-CAM highlights the regions of an image that are most important for the model's prediction:

- **Red/Yellow areas**: High importance - the model focuses heavily on these regions
- **Blue/Green areas**: Low importance - less relevant for the prediction

This helps verify that the model is making decisions based on relevant features (ball shape, texture) rather than background artifacts.

See [dashboard/README.md](dashboard/README.md) for more details.

## API Endpoints

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint - API info and status |
| `/health` | GET | Health check endpoint |
| `/categories` | GET | List all supported ball categories |

### Predictions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload an image and get a prediction |
| `/predictions` | GET | Get prediction history (paginated) |
| `/predictions/{id}` | GET | Get a specific prediction by ID |
| `/predictions/{id}` | DELETE | Delete a specific prediction |
| `/predictions` | DELETE | Clear all predictions |

### Statistics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | Get prediction statistics |

## Usage Examples

### Predict an Image

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "img=@basketball.jpg"
```

Response:
```json
{
  "id": 1,
  "filename": "basketball.jpg",
  "predicted_label": "basketball",
  "confidence": 0.95,
  "all_scores": {
    "american_football": 0.01,
    "baseball": 0.02,
    "basketball": 0.95,
    ...
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Get Prediction History

```bash
curl "http://localhost:8000/predictions?skip=0&limit=10"
```

### Get Statistics

```bash
curl "http://localhost:8000/stats"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to the trained model file | `../model.keras` |
| `DATABASE_URL` | Database connection URL | `sqlite:///./predictions.db` |
| `PORT` | Port to run the API on | `8000` |

## Database

The API supports both SQLite (default) and PostgreSQL databases.

### SQLite (Development)
```bash
export DATABASE_URL=sqlite:///./predictions.db
```

### PostgreSQL (Production)
```bash
export DATABASE_URL=postgresql://user:password@host:5432/dbname
```

## Docker Compose Services

The `docker-compose.yml` includes:

| Service | Port | Description |
|---------|------|-------------|
| **api** | 8000 | FastAPI REST application |
| **dashboard** | 8501 | Streamlit Grad-CAM dashboard |
| **db** | 5432 | PostgreSQL database |

## Kubernetes Deployment

The `kubernetes/deployment.yaml` includes:

- Deployment with 2 replicas
- LoadBalancer service
- Secrets for database configuration
- PersistentVolumeClaim for data storage

To deploy:
```bash
kubectl apply -f ../kubernetes/deployment.yaml
```

## API Documentation

Once the API is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Model Information

The API expects a Keras model trained on 64x64 RGB images with 15 output classes corresponding to the ball categories. The model should output softmax probabilities for each class.

## Project Structure

```
inference/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies (API)
├── Dockerfile           # Docker image for API
├── docker-compose.yml   # Multi-service orchestration
├── README.md            # This file
├── model/               # Trained model directory
│   └── sports-ball-classification/
│       └── model.keras
└── dashboard/           # Streamlit Grad-CAM dashboard
    ├── app.py           # Dashboard application
    ├── gradcam.py       # Grad-CAM implementation
    ├── requirements.txt # Python dependencies (dashboard)
    ├── Dockerfile       # Docker image for dashboard
    └── README.md        # Dashboard documentation
```

## License

This project is part of an MLOps exam assignment for Howest.