import argparse
import logging
import os
from glob import glob

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function of the script - resizes images to 64x64 for ML training."""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output_data", type=str, help="path to output data")
    parser.add_argument(
        "--size", type=int, default=64, help="target image size (default: 64)"
    )
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("Input data:", args.data)
    print("Output folder:", args.output_data)

    output_dir = args.output_data
    size = (args.size, args.size)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files (jpg, jpeg, png)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.data, ext)))

    print(f"Found {len(image_files)} images to process")

    processed_count = 0
    error_count = 0

    for file in image_files:
        try:
            img = Image.open(file)
            # Convert to RGB (in case of RGBA or other formats)
            img = img.convert("RGB")
            img_resized = img.resize(size, Image.Resampling.LANCZOS)

            # Save the resized image to the output directory
            output_file = os.path.join(output_dir, os.path.basename(file))
            # Ensure .jpg extension for consistency
            if not output_file.lower().endswith(".jpg"):
                output_file = os.path.splitext(output_file)[0] + ".jpg"

            img_resized.save(output_file, "JPEG", quality=95)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")
            error_count += 1

    print(
        f"Processing complete: {processed_count} images resized, {error_count} errors"
    )


if __name__ == "__main__":
    main()
