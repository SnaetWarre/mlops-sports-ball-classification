import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database model for storing predictions
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    predicted_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    all_scores = Column(String, nullable=False)  # JSON string of all scores
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic models for API responses
class PredictionResponse(BaseModel):
    id: int
    filename: Optional[str]
    predicted_label: str
    confidence: float
    all_scores: dict
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    total: int
    predictions: List[PredictionResponse]


# Sports ball categories (15 classes) - must match training order
BALL_CATEGORIES: List[str] = [
    "american_football",
    "baseball",
    "basketball",
    "billiard_ball",
    "bowling_ball",
    "cricket_ball",
    "football",
    "golf_ball",
    "hockey_ball",
    "hockey_puck",
    "rugby_ball",
    "shuttlecock",
    "table_tennis_ball",
    "tennis_ball",
    "volleyball",
]

# Global model variable
model = None


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_model():
    """Load the TensorFlow model."""
    global model

    # Resolve model path from environment or default location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.environ.get("MODEL_PATH") or os.path.join(base_dir, "model.keras")

    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            # Try alternate path with model subfolder
            alt_path = os.path.join(base_dir, "model", "sports-ball-cnn", "model.keras")
            if os.path.exists(alt_path):
                logger.info(f"Loading model from alternate path: {alt_path}")
                model = tf.keras.models.load_model(alt_path)
                logger.info("Model loaded successfully from alternate path")
            else:
                logger.warning(
                    f"Model not found at {model_path} or {alt_path}. "
                    "Set MODEL_PATH environment variable to specify model location."
                )
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting up Sports Ball Classification API...")
    Base.metadata.create_all(bind=engine)
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Sports Ball Classification API",
    description="API for classifying sports balls using a CNN model. Supports 15 different ball types.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Root endpoint - health check."""
    return {
        "message": "Sports Ball Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "categories": BALL_CATEGORIES,
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": "connected",
    }


@app.get("/categories")
def get_categories():
    """Get list of supported ball categories."""
    return {"categories": BALL_CATEGORIES, "count": len(BALL_CATEGORIES)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    img: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an image and get a prediction for the type of sports ball.

    The image will be resized to 64x64 pixels and processed by the CNN model.
    The prediction is stored in the database for history tracking.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on the server. Please check server logs.",
        )

    try:
        # Read and preprocess the image
        original_image = Image.open(img.file).convert("RGB")
        original_image = original_image.resize((64, 64))

        # Convert to numpy array (raw pixel values, not normalized)
        img_array = np.array(original_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        probs = np.asarray(predictions).squeeze()

        # Get the predicted class
        if probs.ndim == 0:
            label_idx = int(np.round(probs))
            confidence = float(probs)
        else:
            label_idx = int(np.argmax(probs))
            confidence = float(probs[label_idx])

        predicted_label = (
            BALL_CATEGORIES[label_idx]
            if 0 <= label_idx < len(BALL_CATEGORIES)
            else f"unknown_{label_idx}"
        )

        # Create scores dictionary
        if probs.ndim == 0:
            all_scores = {BALL_CATEGORIES[0]: float(probs)}
        else:
            all_scores = {
                BALL_CATEGORIES[i]: float(probs[i])
                for i in range(min(len(BALL_CATEGORIES), len(probs)))
            }

        # Store prediction in database
        import json

        db_record = PredictionRecord(
            filename=img.filename,
            predicted_label=predicted_label,
            confidence=confidence,
            all_scores=json.dumps(all_scores),
            created_at=datetime.utcnow(),
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return PredictionResponse(
            id=db_record.id,
            filename=db_record.filename,
            predicted_label=predicted_label,
            confidence=confidence,
            all_scores=all_scores,
            created_at=db_record.created_at,
        )

    except Exception as e:
        logger.exception("Failed to process image")
        raise HTTPException(
            status_code=400, detail=f"Failed to process image: {str(e)}"
        )


@app.get("/predictions", response_model=PredictionHistoryResponse)
def get_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Get prediction history from the database.

    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return (default: 100)
    """
    import json

    total = db.query(PredictionRecord).count()
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    predictions = [
        PredictionResponse(
            id=r.id,
            filename=r.filename,
            predicted_label=r.predicted_label,
            confidence=r.confidence,
            all_scores=json.loads(r.all_scores),
            created_at=r.created_at,
        )
        for r in records
    ]

    return PredictionHistoryResponse(total=total, predictions=predictions)


@app.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a specific prediction by ID."""
    import json

    record = (
        db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return PredictionResponse(
        id=record.id,
        filename=record.filename,
        predicted_label=record.predicted_label,
        confidence=record.confidence,
        all_scores=json.loads(record.all_scores),
        created_at=record.created_at,
    )


@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Delete a specific prediction by ID."""
    record = (
        db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    db.delete(record)
    db.commit()

    return {"message": f"Prediction {prediction_id} deleted successfully"}


@app.delete("/predictions")
def clear_predictions(db: Session = Depends(get_db)):
    """Clear all predictions from the database."""
    count = db.query(PredictionRecord).count()
    db.query(PredictionRecord).delete()
    db.commit()

    return {"message": f"Deleted {count} predictions"}


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get statistics about predictions."""
    from sqlalchemy import func

    total = db.query(PredictionRecord).count()

    # Get count per category
    category_stats = (
        db.query(
            PredictionRecord.predicted_label,
            func.count(PredictionRecord.id).label("count"),
        )
        .group_by(PredictionRecord.predicted_label)
        .all()
    )

    # Get average confidence
    avg_confidence = db.query(func.avg(PredictionRecord.confidence)).scalar()

    return {
        "total_predictions": total,
        "average_confidence": float(avg_confidence) if avg_confidence else 0.0,
        "predictions_per_category": {stat[0]: stat[1] for stat in category_stats},
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
