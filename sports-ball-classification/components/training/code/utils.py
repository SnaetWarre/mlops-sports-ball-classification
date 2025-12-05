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
# Order matters for label matching - longer names first to avoid partial matches
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

# Sorted by length (longest first) to avoid partial matching issues
# e.g., "tennis" matching before "table_tennis"
BALL_CATEGORIES_BY_LENGTH = sorted(BALL_CATEGORIES, key=len, reverse=True)


def getTargets(filepaths: List[str]) -> List[str]:
    """
    Extract labels from file paths.
    Assumes filename format: <ball_type>_<number>.jpg
    e.g., 'tennis_ball_123.jpg' -> 'tennis_ball'
          'american_football_45.jpg' -> 'american_football'
          'basketball_99.jpg' -> 'basketball'

    Handles multi-word ball types like 'american_football', 'table_tennis_ball', etc.
    """
    labels = []
    for fp in filepaths:
        filename = fp.split("/")[-1]  # Get only the filename
        # Remove extension
        name_without_ext = filename.rsplit(".", 1)[0].lower()

        # Find which ball category this belongs to
        # Check longest names first to avoid partial matches
        found_label = None
        for category in BALL_CATEGORIES_BY_LENGTH:
            # Check if filename starts with the category name
            # The category should be followed by underscore and a number
            if (
                name_without_ext.startswith(category + "_")
                or name_without_ext == category
            ):
                found_label = category
                break
            # Also check without _ball or _puck suffix for flexibility
            category_base = category.replace("_ball", "").replace("_puck", "")
            if name_without_ext.startswith(category_base + "_"):
                # Verify this is actually the right category
                # by checking if the next part is a number
                remainder = name_without_ext[len(category_base) + 1 :]
                # If remainder starts with a digit, this is the match
                if remainder and remainder.split("_")[0].isdigit():
                    found_label = category
                    break

        if found_label is None:
            # Fallback: try to find any category name in the filename
            for category in BALL_CATEGORIES_BY_LENGTH:
                if category in name_without_ext:
                    found_label = category
                    break

        if found_label is None:
            # Last resort fallback
            parts = name_without_ext.rsplit("_", 1)
            found_label = parts[0] if len(parts) > 1 else name_without_ext
            print(
                f"WARNING: Could not match '{filename}' to known category, using: {found_label}"
            )

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
    print(f"Number of classes: {len(LABELS)}")

    # Print label distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining label distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    return LABELS, y_train_1h, y_test_1h


def getFeatures(filepaths: List[str]) -> np.ndarray:
    """
    Load images from file paths and return as normalized numpy array.

    IMPORTANT: Normalizes pixel values to [0, 1] range for proper training!
    """
    images = []
    for imagePath in filepaths:
        image = Image.open(imagePath).convert("RGB")
        image = np.array(image, dtype=np.float32)
        # CRITICAL: Normalize to [0, 1] range
        image = image / 255.0
        images.append(image)

    images_array = np.array(images)
    print(
        f"Loaded {len(images)} images, shape: {images_array.shape}, "
        f"dtype: {images_array.dtype}, range: [{images_array.min():.2f}, {images_array.max():.2f}]"
    )

    return images_array


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
