# Sports Ball Classification API

A FastAPI-based REST API for classifying sports balls using a trained CNN model. This API supports 15 different types of sports balls and includes database integration for storing prediction history.

## Features

- 🏀 **15 Ball Categories**: american_football, baseball, basketball, billiard_ball, bowling_ball, cricket_ball, football, golf_ball, hockey_ball, hockey_puck, rugby_ball, shuttlecock, table_tennis_ball, tennis_ball, volleyball
- 📊 **Prediction History**: All predictions are stored in a database for analytics and history tracking
- 🐳 **Docker Ready**: Includes Dockerfile and docker-compose for easy deployment
- ☸️ **Kubernetes Ready**: Deployment manifests included
- 📈 **Statistics Endpoint**: Get insights about prediction patterns

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A trained model file (`model.keras`)

### Running with Docker Compose

1. Place your trained model in the `model/sports-ball-cnn/` directory:
   ```
   inference/
   ├── model/
   │   └── sports-ball-cnn/
   │       └── model.keras
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Access the API at `http://localhost:8000`

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

- **api**: The FastAPI application
- **db**: PostgreSQL database for production use

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

## License

This project is part of an MLOps exam assignment.