from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Sports ball categories (15 classes)
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


def getTargets(filepaths: List[str]) -> List[str]:
    """
    Extract labels from file paths.
    Assumes filename format: <ball_type>_<number>.jpg
    e.g., 'tennis_ball_123.jpg' -> 'tennis'
          'american_football_45.jpg' -> 'american'

    We need to handle multi-word ball types like 'american_football', 'hockey_puck', etc.
    """
    labels = []
    for fp in filepaths:
        filename = fp.split("/")[-1]  # Get only the filename
        # Remove extension
        name_without_ext = filename.rsplit(".", 1)[0]

        # Find which ball category this belongs to by checking prefixes
        found_label = None
        for category in BALL_CATEGORIES:
            # Check if filename starts with the category name (before the number)
            if name_without_ext.startswith(
                category.replace("_ball", "").replace("_puck", "")
            ):
                found_label = category
                break

        if found_label is None:
            # Fallback: use the part before the last underscore and number
            parts = name_without_ext.rsplit("_", 1)
            found_label = parts[0] if len(parts) > 1 else name_without_ext

        labels.append(found_label)

    return labels


def encodeLabels(
    y_train: List, y_test: List
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode string labels to one-hot encoded vectors.
    """
    label_encoder = LabelEncoder()
    y_train_labels = label_encoder.fit_transform(y_train)
    y_test_labels = label_encoder.transform(y_test)

    y_train_1h = to_categorical(y_train_labels)
    y_test_1h = to_categorical(y_test_labels)

    LABELS = label_encoder.classes_
    print(f"Labels: {LABELS}")
    print(f"Encoded values: {label_encoder.transform(LABELS)}")

    return LABELS, y_train_1h, y_test_1h


def getFeatures(filepaths: List[str]) -> np.ndarray:
    """
    Load images from file paths and return as numpy array.
    """
    images = []
    for imagePath in filepaths:
        image = Image.open(imagePath).convert("RGB")
        image = np.array(image)
        images.append(image)
    return np.array(images)


def buildModel(inputShape: tuple, classes: int) -> Sequential:
    """
    Build a CNN model for image classification.

    Architecture:
    - 3 convolutional blocks with increasing filter sizes (32 -> 64 -> 128)
    - Batch normalization and dropout for regularization
    - Fully connected layer with 512 units
    - Softmax output layer for multi-class classification

    Args:
        inputShape: Tuple of (height, width, channels) - typically (64, 64, 3)
        classes: Number of output classes (15 for sports balls)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    height, width, depth = inputShape
    inputShape = (height, width, depth)
    chanDim = -1

    # CONV => RELU => POOL layer set
    # First CONV layer has 32 filters of size 3x3
    model.add(
        Conv2D(32, (3, 3), padding="same", name="conv_32_1", input_shape=inputShape)
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL layer set
    # Increase total number of filters (from 32 to 64)
    model.add(Conv2D(64, (3, 3), padding="same", name="conv_64_1"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same", name="conv_64_2"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL layer set
    # Total number of filters doubled again (128)
    model.add(Conv2D(128, (3, 3), padding="same", name="conv_128_1"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same", name="conv_128_2"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same", name="conv_128_3"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected layer (FC) => RELU
    model.add(Flatten())
    model.add(Dense(512, name="fc_1"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Softmax classifier for multi-class output
    model.add(Dense(classes, name="output"))
    model.add(Activation("softmax"))

    return model
