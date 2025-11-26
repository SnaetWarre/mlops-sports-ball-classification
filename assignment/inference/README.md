# Animal Classification Inference API

This folder contains a FastAPI application that loads a Keras model and exposes endpoints to classify uploaded animal images (Cat, Dog, or Panda).

## Quick Start

### Running with Docker Compose (Recommended)

```bash
cd inference
docker-compose up --build -d
```

The API will be available at `http://localhost:8000`

- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/

To view logs:
```bash
docker-compose logs -f api
```

To stop:
```bash
docker-compose down
```

### Running Locally (with conda)

```bash
conda activate mlops
pip install -r requirements.txt
python main.py
# Open http://localhost:8000/docs to interact with the API
```

## Configuration

- **MODEL_PATH**: Environment variable pointing to the Keras model file. Default: `/app/animal-classification/INPUT_model_path/animal-cnn/model.keras` (set in `docker-compose.yml`)
- **PORT**: Server port (default: 8000)
- **HF Hub fallback** (used when `MODEL_PATH` does not exist):
  - **HF_REPO_ID**: your Hugging Face model repo id, e.g. `<your-username>/masterclass-2025`
  - **HF_MODEL_FILENAME**: model file name in the repo (default: `model.keras`)
  - **HF_REVISION**: optional tag/branch/commit to pin a version (default: `main`)

Example docker-compose env:
```yaml
environment:
  - PORT=8000
  - MODEL_PATH=/app/animal-classification/INPUT_model_path/animal-cnn/model.keras
  - HF_REPO_ID=<your-username>/masterclass-2025
  - HF_MODEL_FILENAME=model.keras
  - HF_REVISION=main
```

## API Endpoints

- `GET /` - Health check, returns `{"hello": "world"}`
- `POST /upload/image` - Upload an image file for classification. Returns predicted label and confidence scores.

Minimal curl example:
```bash
curl -X POST http://localhost:8000/upload/image \
  -H "Accept: application/json" \
  -F "img=@/path/to/image.jpg"
```

Sample response:
```json
{
  "label": "Dog",
  "scores": [0.01, 0.97, 0.02]
}
```

## Requirements

- **TensorFlow**: 2.16+ (includes Keras 3.x support for `.keras` model format)
- **Python**: 3.10
- **Docker**: Required for containerized deployment

## Notes

- The API expects images that can be opened by Pillow
- Images are automatically resized to (64, 64) before prediction
- Supported animal classes: Cat, Dog, Panda
- Model is loaded from the path specified by `MODEL_PATH` environment variable
