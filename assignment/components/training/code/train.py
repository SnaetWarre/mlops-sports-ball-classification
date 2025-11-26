import argparse
import os
import random
from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# This time we will need our Tensorflow Keras libraries, as we will be working with the AI training now
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Important to load in the utils as well!
from utils import *

### HARDCODED VARIABLES FOR NOW
### TODO for the students:
### Make sure to adapt the ArgumentParser on line 31 to include these parameters
### You can base your answer on the lines that are already there

SEED = 42
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 32
PATIENCE = 11
model_name = "animal-cnn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_folder",
        type=str,
        dest="training_folder",
        help="training folder mounting point",
    )
    parser.add_argument(
        "--testing_folder",
        type=str,
        dest="testing_folder",
        help="testing folder mounting point",
    )
    parser.add_argument(
        "--output_folder", type=str, dest="output_folder", help="Output folder"
    )
    parser.add_argument(
        "--epochs", type=int, dest="epochs", help="The amount of Epochs to train"
    )
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    training_folder = args.training_folder
    print("Training folder:", training_folder)

    testing_folder = args.testing_folder
    print("Testing folder:", testing_folder)

    output_folder = args.output_folder
    print("Testing folder:", output_folder)

    MAX_EPOCHS = args.epochs

    # As we're mounting the training_folder and testing_folder onto the `/mnt/data` directories, we can load in the images by using glob.
    training_paths = glob(training_folder + "/*.jpg", recursive=True)
    testing_paths = glob(testing_folder + "/*.jpg", recursive=True)

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))

    # Make sure to shuffle in the same way as I'm doing everything
    random.seed(SEED)
    random.shuffle(training_paths)
    random.seed(SEED)
    random.shuffle(testing_paths)

    print(training_paths[:3])  # Examples
    print(testing_paths[:3])  # Examples

    # Parse to Features and Targets for both Training and Testing. Refer to the Utils package for more information
    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)

    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print("Shapes:")
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    # Make sure the data is one-hot-encoded
    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print("One Hot Shapes:")

    print(y_train.shape)
    print(y_test.shape)

    # Create an output directory where our AI model will be saved to.
    # Everything inside the `outputs` directory will be logged and kept aside for later usage.
    model_directory = os.path.join(output_folder, model_name)
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, "model.keras")

    # Save the best model, not the last
    cb_save_best_model = keras.callbacks.ModelCheckpoint(
        filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1
    )

    # Early stop when the val_los isn't improving for PATIENCE epochs
    cb_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=1, restore_best_weights=True
    )

    # Use ExponentialDecay learning rate schedule
    # Note: Cannot use ReduceLROnPlateau with LearningRateSchedule - they conflict
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=MAX_EPOCHS,
        decay_rate=0.5,  # e.g. halve the LR every `decay_steps`
        staircase=True,  # if you prefer discrete drops
    )

    # Plug the schedule into the optimizer
    opt = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.0,  # if you need momentum
        nesterov=False,
    )

    model = buildModel(
        (64, 64, 3), 3
    )  # Create the AI model as defined in the utils script.

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Construct & initialize the image data generator for data augmentation
    # Image augmentation allows us to construct “additional” training data from our existing training data
    # by randomly rotating, shifting, shearing, zooming, and flipping. This is to avoid overfitting.
    # It also allows us to fit AI models using a Generator, so we don't need to capture the whole dataset in memory at once.
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # train the network
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[cb_save_best_model, cb_early_stop],
    )

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print(
        classification_report(
            y_test.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=["cats", "dogs", "panda"],
        )
    )  # Give the target names to easier refer to them.
    # If you want, you can enter the target names as a parameter as well, in case you ever adapt your AI model to more animals.

    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(cf_matrix)

    ### TODO for students (for later)
    ### Find a way to log more information to the Run context.

    # Save the confusion matrix to the outputs.
    np.save(os.path.join(output_folder, "confusion_matrix.npy"), cf_matrix)

    print("DONE TRAINING")


if __name__ == "__main__":
    main()
