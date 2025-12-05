"""
Sports Ball Classification Dashboard with Grad-CAM Visualization

A Streamlit dashboard that provides:
- Image upload for sports ball classification
- Prediction results with confidence scores
- Grad-CAM heatmap visualization showing what the model is looking at
- Side-by-side comparison of original image and heatmap overlay
"""

import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from gradcam import GradCAM, overlay_heatmap
from PIL import Image

# Sports ball categories (must match training order)
BALL_CATEGORIES = [
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

# Emoji mapping for visual appeal
BALL_EMOJIS = {
    "american_football": "🏈",
    "baseball": "⚾",
    "basketball": "🏀",
    "billiard_ball": "🎱",
    "bowling_ball": "🎳",
    "cricket_ball": "🏏",
    "football": "⚽",
    "golf_ball": "⛳",
    "hockey_ball": "🏑",
    "hockey_puck": "🥅",
    "rugby_ball": "🏉",
    "shuttlecock": "🏸",
    "table_tennis_ball": "🏓",
    "tennis_ball": "🎾",
    "volleyball": "🏐",
}


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    # Try multiple possible model paths
    possible_paths = [
        os.environ.get("MODEL_PATH", ""),
        "/app/model/sports-ball-classification/model.keras",
        "../model/sports-ball-classification/model.keras",
        "../../model/sports-ball-classification/model.keras",
        "../model.keras",
        "./model.keras",
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                st.sidebar.success(f"✅ Model loaded from: {path}")
                return model
            except Exception as e:
                st.sidebar.warning(f"Failed to load from {path}: {e}")
                continue

    st.sidebar.error("❌ Model not found! Please ensure model.keras is available.")
    return None


def preprocess_image(image: Image.Image, target_size=(64, 64)) -> np.ndarray:
    """Preprocess image for model prediction."""
    # Resize to model's expected input size
    image = image.resize(target_size)
    # Convert to numpy array and normalize to [0, 1] range
    # CRITICAL: Must match training normalization!
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_prediction(model, img_array: np.ndarray) -> tuple:
    """Get model prediction and confidence scores."""
    predictions = model.predict(img_array, verbose=0)
    probs = np.asarray(predictions).squeeze()

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

    return predicted_label, confidence, all_scores, label_idx


def find_last_conv_layer(model) -> str:
    """Find the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Fallback to known layer name from the architecture
    return "conv_128_3"


def main():
    # Page configuration
    st.set_page_config(
        page_title="🏀 Sports Ball Classifier",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        .confidence-high { color: #00c853; }
        .confidence-medium { color: #ffc107; }
        .confidence-low { color: #ff5252; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        '<h1 class="main-header">🏀 Sports Ball Classification with Grad-CAM</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
        Upload an image of a sports ball to classify it and see what the AI is looking at!
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Load model
    model = load_model()

    if model is None:
        st.error(
            "🚨 Model could not be loaded. Please check the model path and try again."
        )
        st.stop()

    # Find last conv layer for Grad-CAM
    last_conv_layer = find_last_conv_layer(model)
    st.sidebar.info(f"🔍 Using layer: `{last_conv_layer}` for Grad-CAM")

    # Grad-CAM settings
    st.sidebar.subheader("Grad-CAM Settings")
    heatmap_intensity = st.sidebar.slider(
        "Heatmap Intensity", min_value=0.1, max_value=1.0, value=0.4, step=0.1
    )
    colormap = st.sidebar.selectbox(
        "Colormap", ["jet", "viridis", "plasma", "inferno", "magma"], index=0
    )

    # Supported categories
    st.sidebar.subheader("📋 Supported Categories")
    for cat in BALL_CATEGORIES:
        emoji = BALL_EMOJIS.get(cat, "🔵")
        st.sidebar.text(f"{emoji} {cat.replace('_', ' ').title()}")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image of a sports ball...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear image of a sports ball for classification",
        )

        # Sample images option
        use_sample = st.checkbox("🎲 Use sample image for demo")

    if uploaded_file is not None or use_sample:
        try:
            if use_sample:
                # Create a simple sample image (orange circle for basketball demo)
                sample_img = Image.new("RGB", (64, 64), color=(255, 140, 0))
                original_image = sample_img
                st.info("📌 Using generated sample image (orange square)")
            else:
                # Load uploaded image
                original_image = Image.open(uploaded_file).convert("RGB")

            with col1:
                st.image(
                    original_image,
                    caption="Original Image",
                    use_container_width=True,
                )

            # Preprocess image
            img_array = preprocess_image(original_image)

            # Get prediction
            predicted_label, confidence, all_scores, label_idx = get_prediction(
                model, img_array
            )

            # Generate Grad-CAM heatmap
            with st.spinner("🔥 Generating Grad-CAM visualization..."):
                gradcam = GradCAM(model, last_conv_layer)
                heatmap = gradcam.compute_heatmap(img_array, label_idx)

                # Resize original image to 64x64 for overlay
                resized_original = original_image.resize((64, 64))
                original_array = np.array(resized_original)

                # Create overlay
                overlay_image, heatmap_colored = overlay_heatmap(
                    heatmap, original_array, alpha=heatmap_intensity, colormap=colormap
                )

            with col2:
                st.subheader("🔥 Grad-CAM Heatmap")

                # Show heatmap overlay
                st.image(
                    overlay_image,
                    caption="What the model is looking at",
                    use_container_width=True,
                )

                # Show pure heatmap
                with st.expander("🎨 View raw heatmap"):
                    st.image(
                        heatmap_colored,
                        caption="Raw Grad-CAM Heatmap",
                        use_container_width=True,
                    )

            # Prediction results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")

            emoji = BALL_EMOJIS.get(predicted_label, "🔵")

            # Confidence color coding
            if confidence >= 0.8:
                conf_class = "confidence-high"
                conf_label = "High Confidence"
            elif confidence >= 0.5:
                conf_class = "confidence-medium"
                conf_label = "Medium Confidence"
            else:
                conf_class = "confidence-low"
                conf_label = "Low Confidence"

            # Main prediction display
            pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])

            with pred_col2:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2rem;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 15px; color: white; margin: 1rem 0;">
                        <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                        <h2 style="margin: 0.5rem 0;">{predicted_label.replace("_", " ").title()}</h2>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                            Confidence: <strong>{confidence * 100:.1f}%</strong>
                        </p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">{conf_label}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Confidence scores bar chart
            st.subheader("📊 All Confidence Scores")

            # Sort scores by value
            sorted_scores = dict(
                sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            )

            # Create bar chart data
            chart_data = {
                "Category": [
                    f"{BALL_EMOJIS.get(k, '🔵')} {k.replace('_', ' ').title()}"
                    for k in sorted_scores.keys()
                ],
                "Confidence": [v * 100 for v in sorted_scores.values()],
            }

            # Display as horizontal bar chart
            import pandas as pd

            df = pd.DataFrame(chart_data)
            st.bar_chart(df.set_index("Category")["Confidence"], horizontal=True)

            # Detailed scores table
            with st.expander("📋 View detailed scores"):
                detail_df = pd.DataFrame(
                    {
                        "Category": [
                            k.replace("_", " ").title() for k in sorted_scores.keys()
                        ],
                        "Emoji": [
                            BALL_EMOJIS.get(k, "🔵") for k in sorted_scores.keys()
                        ],
                        "Confidence (%)": [
                            f"{v * 100:.2f}" for v in sorted_scores.values()
                        ],
                        "Raw Score": [f"{v:.6f}" for v in sorted_scores.values()],
                    }
                )
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

            # Explanation
            st.markdown("---")
            st.subheader("🧠 How Grad-CAM Works")
            st.markdown(
                """
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainability
                technique that highlights the regions of an image that are most important for
                the model's prediction.

                - **Red/Yellow areas**: High importance - the model focuses heavily on these regions
                - **Blue/Green areas**: Low importance - less relevant for the prediction
                - **Overlay**: Combines the heatmap with the original image for easy interpretation

                This helps us understand *what* the model is looking at and verify it's making
                decisions based on relevant features (like the ball's shape and texture) rather
                than background artifacts.
                """
            )

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)

    else:
        # No image uploaded - show instructions
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem; background: #f0f2f6;
                        border-radius: 15px; margin: 2rem 0;">
                <h2>👆 Upload an image to get started!</h2>
                <p style="color: #666;">
                    Drag and drop an image of a sports ball, or click the upload button above.
                </p>
                <p style="color: #888; font-size: 0.9rem;">
                    Supported formats: JPG, JPEG, PNG, WEBP
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show example of what to expect
        st.subheader("📸 What to expect")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            st.markdown(
                """
                #### 1️⃣ Upload
                Upload a clear image of any sports ball from the 15 supported categories.
                """
            )

        with exp_col2:
            st.markdown(
                """
                #### 2️⃣ Classify
                Our CNN model analyzes the image and predicts the ball type with confidence scores.
                """
            )

        with exp_col3:
            st.markdown(
                """
                #### 3️⃣ Explain
                Grad-CAM shows you exactly which parts of the image influenced the prediction.
                """
            )


if __name__ == "__main__":
    main()
