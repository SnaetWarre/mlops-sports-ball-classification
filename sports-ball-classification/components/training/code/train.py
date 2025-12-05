import argparse
import os
import random
from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import BALL_CATEGORIES, buildModel, encodeLabels, getFeatures, getTargets

# Hyperparameters
SEED = 42
INITIAL_LEARNING_RATE = 0.001  # Lower LR for Adam
BATCH_SIZE = 32
PATIENCE = 10
MODEL_NAME = "sports-ball-cnn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_folder",
        type=str,
        dest="training_folder",
        help="Training folder mounting point",
    )
    parser.add_argument(
        "--testing_folder",
        type=str,
        dest="testing_folder",
        help="Testing folder mounting point",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        help="Output folder for model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        default=30,  # Increased from 10 to 30 for better convergence
        help="The number of epochs to train",
    )
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    training_folder = args.training_folder
    print("Training folder:", training_folder)

    testing_folder = args.testing_folder
    print("Testing folder:", testing_folder)

    output_folder = args.output_folder
    print("Output folder:", output_folder)

    MAX_EPOCHS = args.epochs
    print("Max epochs:", MAX_EPOCHS)

    # Load image paths
    training_paths = glob(os.path.join(training_folder, "*.jpg"), recursive=True)
    testing_paths = glob(os.path.join(testing_folder, "*.jpg"), recursive=True)

    print(f"Training samples: {len(training_paths)}")
    print(f"Testing samples: {len(testing_paths)}")

    # Shuffle with fixed seed for reproducibility
    random.seed(SEED)
    random.shuffle(training_paths)
    random.seed(SEED)
    random.shuffle(testing_paths)

    print("Sample training paths:", training_paths[:3])
    print("Sample testing paths:", testing_paths[:3])

    # Extract features (images) and targets (labels)
    print("\nLoading training data...")
    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)

    print("Loading testing data...")
    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"X_test: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_train: {len(y_train)} labels")
    print(f"y_test: {len(y_test)} labels")

    # Verify normalization
    print(f"\nX_train value range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"X_test value range: [{X_test.min():.4f}, {X_test.max():.4f}]")

    # One-hot encode labels
    LABELS, y_train_encoded, y_test_encoded = encodeLabels(y_train, y_test)
    num_classes = len(LABELS)

    print(f"\nNumber of classes: {num_classes}")
    print(f"y_train shape: {y_train_encoded.shape}")
    print(f"y_test shape: {y_test_encoded.shape}")

    # Create output directory
    model_directory = os.path.join(output_folder, MODEL_NAME)
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, "model.keras")

    # Save labels for inference
    labels_path = os.path.join(model_directory, "labels.txt")
    with open(labels_path, "w") as f:
        for label in LABELS:
            f.write(f"{label}\n")
    print(f"Labels saved to: {labels_path}")

    # Callbacks
    cb_save_best_model = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",  # Changed to monitor accuracy
        save_best_only=True,
        verbose=1,
    )

    cb_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",  # Changed to monitor accuracy
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True,
        mode="max",  # We want to maximize accuracy
    )

    # ReduceLROnPlateau - reduces learning rate when validation loss plateaus
    cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    # Use Adam optimizer - much better for CNN training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE,
    )

    # Build model for 15 classes (or however many we have)
    print(f"\nBuilding CNN model for {num_classes} classes...")
    model = buildModel((64, 64, 3), num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    model.summary()

    # Data augmentation for training
    # More conservative augmentation to avoid distorting ball shapes too much
    aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Calculate steps per epoch
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    print(f"\nSteps per epoch: {steps_per_epoch}")

    # Train the model
    print("\n[INFO] Training the network...")
    history = model.fit(
        aug.flow(X_train, y_train_encoded, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test_encoded),
        steps_per_epoch=steps_per_epoch,
        epochs=MAX_EPOCHS,
        callbacks=[cb_save_best_model, cb_early_stop, cb_reduce_lr],
    )

    # Evaluate the model
    print("\n[INFO] Evaluating network...")
    predictions = model.predict(X_test, batch_size=32)

    # Print final metrics
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    best_val_acc = max(history.history["val_accuracy"])

    print(f"\n=== Training Results ===")
    print(f"Final training accuracy: {final_train_acc * 100:.2f}%")
    print(f"Final validation accuracy: {final_val_acc * 100:.2f}%")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_encoded.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=LABELS,
        )
    )

    # Confusion matrix
    cf_matrix = confusion_matrix(
        y_test_encoded.argmax(axis=1),
        predictions.argmax(axis=1),
    )
    print("\nConfusion Matrix:")
    print(cf_matrix)

    # Save confusion matrix
    np.save(os.path.join(output_folder, "confusion_matrix.npy"), cf_matrix)

    # Save training history
    history_path = os.path.join(output_folder, "training_history.npy")
    np.save(history_path, history.history)

    print(f"\n[INFO] Model saved to: {model_path}")
    print(f"[INFO] Confusion matrix saved to: {output_folder}/confusion_matrix.npy")
    print("[INFO] DONE TRAINING")


if __name__ == "__main__":
    main()
