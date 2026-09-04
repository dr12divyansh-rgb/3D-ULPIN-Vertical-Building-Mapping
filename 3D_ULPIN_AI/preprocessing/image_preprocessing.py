from pathlib import Path
from PIL import Image
import numpy as np


# Supported image formats
SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


def check_image(image_path):
    """
    Check whether an image exists and can be opened.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}"
        )

    try:
        image = Image.open(image_path)
        image.verify()

    except Exception as error:
        raise ValueError(
            f"Invalid or corrupted image: {error}"
        )

    return True


def get_image_information(image_path):
    """
    Get basic information about an image.
    """

    image_path = Path(image_path)

    check_image(image_path)

    image = Image.open(image_path)

    information = {
        "file_name": image_path.name,
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode
    }

    return information


def load_image(image_path):
    """
    Load an image and convert it into a NumPy array.
    """

    image_path = Path(image_path)

    check_image(image_path)

    image = Image.open(image_path).convert("RGB")

    image_array = np.array(image)

    return image_array


def resize_image(image_array, size=(640, 640)):
    """
    Resize an image to the required model input size.
    """

    image = Image.fromarray(image_array)

    resized = image.resize(size)

    return np.array(resized)


def normalize_image(image_array):
    """
    Normalize pixel values from 0-255 to 0-1.
    """

    normalized = image_array.astype(np.float32) / 255.0

    return normalized


if __name__ == "__main__":

    # Find project root
    project_root = Path(__file__).resolve().parent.parent

    # Temporary test image
    image_path = (
        project_root
        / "data"
        / "raw"
        / "drone"
        / "sample.jpg"
    )

    try:

        print("Checking image...\n")

        # Check image
        check_image(image_path)

        print("Image check: OK")

        # Get information
        information = get_image_information(image_path)

        print("\nImage information:")

        for key, value in information.items():
            print(f"{key}: {value}")

        # Load image
        image = load_image(image_path)

        print("\nImage loaded successfully!")
        print("Array shape:", image.shape)

        # Resize
        resized = resize_image(image)

        print("Resized image shape:", resized.shape)

        # Normalize
        normalized = normalize_image(resized)

        print("Normalized image shape:", normalized.shape)
        print(
            "Pixel value range:",
            normalized.min(),
            "to",
            normalized.max()
        )

        print("\nImage preprocessing test successful!")

    except Exception as error:

        print("\nERROR:", error)