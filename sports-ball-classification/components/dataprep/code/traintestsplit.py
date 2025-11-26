import argparse
import math
import os
import random
from glob import glob


def main():
    """Main function - splits multiple datasets into training and testing sets."""

    SEED = 42

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=str, nargs="+", help="All the datasets to combine"
    )
    parser.add_argument(
        "--training_data_output", type=str, help="path to training output data"
    )
    parser.add_argument(
        "--testing_data_output", type=str, help="path to testing output data"
    )
    parser.add_argument(
        "--split_size", type=int, help="Percentage to use as Testing data"
    )
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("Input datasets:", args.datasets)
    print("Training folder:", args.training_data_output)
    print("Testing folder:", args.testing_data_output)
    print("Split size:", args.split_size)

    train_test_split_factor = args.split_size / 100
    datasets = args.datasets

    # Create output directories
    os.makedirs(args.training_data_output, exist_ok=True)
    os.makedirs(args.testing_data_output, exist_ok=True)

    training_datapaths = []
    testing_datapaths = []

    total_train = 0
    total_test = 0

    for dataset in datasets:
        # Find all image files (jpg, jpeg, png)
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        ball_images = []
        for ext in image_extensions:
            ball_images.extend(glob(os.path.join(dataset, ext)))

        dataset_name = os.path.basename(dataset.rstrip("/"))
        print(f"Found {len(ball_images)} images for {dataset_name}")

        if len(ball_images) == 0:
            print(f"WARNING: No images found in {dataset}")
            continue

        # Use the same random seed for reproducibility
        random.seed(SEED)
        random.shuffle(ball_images)

        # Calculate split
        amount_of_test_images = math.ceil(len(ball_images) * train_test_split_factor)

        ball_test_images = ball_images[:amount_of_test_images]
        ball_training_images = ball_images[amount_of_test_images:]

        # Add them all to the tracking lists
        testing_datapaths.extend(ball_test_images)
        training_datapaths.extend(ball_training_images)

        print(
            f"  -> {len(ball_training_images)} training, {len(ball_test_images)} testing"
        )

        # Write the testing data
        for img in ball_test_images:
            with open(img, "rb") as f:
                output_path = os.path.join(
                    args.testing_data_output, os.path.basename(img)
                )
                with open(output_path, "wb") as f2:
                    f2.write(f.read())
            total_test += 1

        # Write the training data
        for img in ball_training_images:
            with open(img, "rb") as f:
                output_path = os.path.join(
                    args.training_data_output, os.path.basename(img)
                )
                with open(output_path, "wb") as f2:
                    f2.write(f.read())
            total_train += 1

    print(f"\n=== Split Complete ===")
    print(f"Total training images: {total_train}")
    print(f"Total testing images: {total_test}")
    print(f"Split ratio: {100 - args.split_size}% train / {args.split_size}% test")


if __name__ == "__main__":
    main()
