"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

This module provides functionality to visualize which parts of an image
the CNN model focuses on when making predictions.

Reference: https://arxiv.org/abs/1610.02391
"""

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN attention.

    Grad-CAM uses the gradients of any target concept flowing into the
    final convolutional layer to produce a coarse localization map
    highlighting important regions in the image for predicting the concept.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        """
        Initialize GradCAM with a model.

        Args:
            model: A trained Keras model
            layer_name: Name of the convolutional layer to use for Grad-CAM.
                       If None, will try to find the last Conv2D layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()

        # Create a model that outputs both the conv layer output and predictions
        self.grad_model = self._build_grad_model()

    def _find_last_conv_layer(self) -> str:
        """Find the name of the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("Could not find any Conv2D layer in the model")

    def _build_grad_model(self) -> tf.keras.Model:
        """Build a model that outputs conv layer activations and predictions."""
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
        )

    def compute_heatmap(
        self, img_array: np.ndarray, pred_index: int = None, eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for an image.

        Args:
            img_array: Preprocessed image array of shape (1, H, W, C)
            pred_index: Index of the class to visualize. If None, uses
                       the predicted class.
            eps: Small constant to avoid division by zero

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1]
        """
        # Cast to float32 for gradient computation
        img_tensor = tf.cast(img_array, tf.float32)

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = self.grad_model(img_tensor)

            # If no class index specified, use the predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            # Get the score for the target class
            class_score = predictions[:, pred_index]

        # Compute gradients of class score with respect to conv layer output
        grads = tape.gradient(class_score, conv_outputs)

        # Compute the mean intensity of the gradient over each feature map
        # This gives us the "importance weights" for each filter
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Get the conv layer output for this image
        conv_outputs = conv_outputs[0]

        # Weight each feature map by its importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to focus only on positive contributions
        heatmap = tf.nn.relu(heatmap)

        # Normalize the heatmap to [0, 1]
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)

        return heatmap.numpy()


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> tuple:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        heatmap: Heatmap array from compute_heatmap() with shape (H, W)
        original_image: Original image as numpy array with shape (H, W, 3)
        alpha: Transparency for the heatmap overlay (0-1)
        colormap: Matplotlib colormap name

    Returns:
        Tuple of (overlay_image, heatmap_colored) as numpy arrays
    """
    # Get original image dimensions
    height, width = original_image.shape[:2]

    # Resize heatmap to match original image size
    heatmap_resized = (
        np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (width, height), resample=Image.BILINEAR
            )
        )
        / 255.0
    )

    # Apply colormap
    colormap_fn = cm.get_cmap(colormap)
    heatmap_colored = colormap_fn(heatmap_resized)

    # Convert to RGB (drop alpha channel from colormap) and scale to 0-255
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # Ensure original image is in the right format
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

    # Blend images
    overlay = (
        (1 - alpha) * original_image.astype(np.float32)
        + alpha * heatmap_colored.astype(np.float32)
    ).astype(np.uint8)

    return overlay, heatmap_colored


def create_gradcam(model: tf.keras.Model) -> GradCAM:
    """
    Factory function to create a GradCAM instance.

    Automatically finds the best convolutional layer for visualization.
    For the sports ball CNN, this will be 'conv_128_3'.

    Args:
        model: Trained Keras model

    Returns:
        GradCAM instance
    """
    # Try to use conv_128_3 (the last conv layer in our sports ball CNN)
    # Fall back to automatic detection if not found
    try:
        model.get_layer("conv_128_3")
        layer_name = "conv_128_3"
    except ValueError:
        layer_name = None  # Will auto-detect

    return GradCAM(model, layer_name=layer_name)
