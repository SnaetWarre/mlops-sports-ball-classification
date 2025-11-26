from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
import logging
from typing import List

import numpy as np
from PIL import Image

import tensorflow as tf
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Animal Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ANIMALS: List[str] = ["Cat", "Dog", "Panda"]


# Resolve model path. Prefer an environment variable for flexibility. As a fallback
# try a model.keras file in the repository root (one level up from this file).
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.environ.get("MODEL_PATH") or os.path.join(base_dir, "model.keras")


model = None
try:
    if os.path.exists(model_path):
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    else:
        # Try Hugging Face Hub fallback when local model is missing
        repo_id = os.environ.get("HF_REPO_ID")
        filename = os.environ.get("HF_MODEL_FILENAME", "model.keras")
        revision = os.environ.get("HF_REVISION")
        if repo_id:
            try:
                logger.info(
                    f"Local model not found. Trying HF Hub: repo_id={repo_id}, filename={filename}, revision={revision}"
                )
                local_dir = os.path.join(base_dir, "hf_model")
                os.makedirs(local_dir, exist_ok=True)
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                    revision=revision,
                    local_dir=local_dir,
                )
                logger.info(f"Downloaded model file to: {downloaded_path}")
                model = tf.keras.models.load_model(downloaded_path)
                logger.info("Model loaded successfully from HF Hub")
            except Exception:
                logger.exception("HF Hub fallback failed")
        if model is None:
            logger.warning(
                "Model not available. Set MODEL_PATH or HF_REPO_ID (+ HF_MODEL_FILENAME) environment variables."
            )
except Exception as e:
    logger.exception("Failed to load model")
    model = None


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post("/upload/image")
async def upload_image(img: UploadFile = File(...)):
    """Accept an uploaded image, resize to (64,64), run model.predict and return the label.

    If the model is not available the endpoint will return 503.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server")

    try:
        # Read image bytes and ensure RGB
        original_image = Image.open(img.file).convert("RGB")
        # Preprocess the image
        original_image = original_image.resize((64, 64))
        # Training used raw pixel values [0-255], NOT normalized to [0-1]
        img_array = np.array(original_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        # predictions might be shape (1, N)
        probs = np.asarray(predictions).squeeze()
        if probs.ndim == 0:
            # Model returned a single value
            label_idx = int(np.round(probs))
        else:
            label_idx = int(np.argmax(probs))

        label = ANIMALS[label_idx] if 0 <= label_idx < len(ANIMALS) else str(label_idx)

        return JSONResponse({"label": label, "scores": probs.tolist()})

    except Exception:
        logger.exception("Failed to process image")
        raise HTTPException(status_code=400, detail="Failed to process image")


if __name__ == "__main__":
    # Run with: python main.py
    # Use Uvicorn as the ASGI server. MODEL_PATH and PORT can be overridden via env vars.
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
