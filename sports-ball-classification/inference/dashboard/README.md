# 🏀 Sports Ball Classification Dashboard

An interactive Streamlit dashboard with **Grad-CAM visualization** for the Sports Ball Classification model.

## ✨ Features

- **Image Upload**: Drag & drop or browse to upload sports ball images
- **Real-time Classification**: Instant predictions across 15 ball categories
- **Grad-CAM Heatmaps**: Visualize what the CNN model is "looking at"
- **Confidence Scores**: Bar chart showing all category probabilities
- **Interactive Settings**: Adjust heatmap intensity and colormap

## 🧠 What is Grad-CAM?

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainability technique that highlights the regions of an image that are most important for the model's prediction.

- **Red/Yellow areas**: High importance - the model focuses heavily on these regions
- **Blue/Green areas**: Low importance - less relevant for the prediction

This helps verify that the model is making decisions based on relevant features (like the ball's shape and texture) rather than background artifacts.

## 🚀 Quick Start

### Using Docker Compose (Recommended)

From the `inference/` directory:

```bash
docker-compose up -d dashboard
```

Then open http://localhost:8501 in your browser.

### Using Docker Directly

```bash
cd inference/dashboard
docker build -t sports-ball-dashboard .
docker run -p 8501:8501 -v ../model:/app/model sports-ball-dashboard
```

### Running Locally (Development)

```bash
cd inference/dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set model path
export MODEL_PATH="../model/sports-ball-classification/model.keras"

# Run Streamlit
streamlit run app.py
```

## 📁 Project Structure

```
dashboard/
├── app.py              # Main Streamlit application
├── gradcam.py          # Grad-CAM implementation
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
└── README.md           # This file
```

## 🎮 Usage

1. **Open the dashboard** at http://localhost:8501
2. **Upload an image** of a sports ball (JPG, PNG, or WEBP)
3. **View the prediction** with confidence scores
4. **Explore the Grad-CAM heatmap** to see what the model focused on
5. **Adjust settings** in the sidebar to customize the visualization

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to the trained model file | `/app/model/sports-ball-classification/model.keras` |
| `STREAMLIT_SERVER_PORT` | Port for the Streamlit server | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Address to bind to | `0.0.0.0` |

### Sidebar Settings

- **Heatmap Intensity**: Controls the transparency of the heatmap overlay (0.1 - 1.0)
- **Colormap**: Choose from jet, viridis, plasma, inferno, or magma

## 🏐 Supported Ball Categories

| Emoji | Category | Emoji | Category |
|-------|----------|-------|----------|
| 🏈 | American Football | 🏑 | Hockey Ball |
| ⚾ | Baseball | 🥅 | Hockey Puck |
| 🏀 | Basketball | 🏉 | Rugby Ball |
| 🎱 | Billiard Ball | 🏸 | Shuttlecock |
| 🎳 | Bowling Ball | 🏓 | Table Tennis Ball |
| 🏏 | Cricket Ball | 🎾 | Tennis Ball |
| ⚽ | Football | 🏐 | Volleyball |
| ⛳ | Golf Ball | | |

## 🔧 Technical Details

### Grad-CAM Implementation

The Grad-CAM implementation uses TensorFlow's `GradientTape` to compute gradients of the predicted class score with respect to the last convolutional layer (`conv_128_3`). The weighted sum of feature maps produces the attention heatmap.

### Model Compatibility

This dashboard is designed to work with the CNN model trained by the Azure ML pipeline. The model should:
- Accept 64x64 RGB images
- Output 15-class softmax probabilities
- Have named convolutional layers (e.g., `conv_128_3`)

## 📝 License

Part of the MLOps Exam Project for Howest.