from pathlib import Path

import numpy as np
from PIL import Image

import torch

from train import UNet


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):
    """
    Load the trained building segmentation model.
    """

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ============================================================
# PREDICT BUILDINGS
# ============================================================

def detect_buildings(image_path, model_path):
    """
    Detect buildings in an image.

    Returns a binary building mask.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Resize for model
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to NumPy
    image_array = np.array(image)

    # Normalize
    image_array = (
        image_array.astype(np.float32)
        / 255.0
    )

    # H x W x C
    # ↓
    # C x H x W
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Add batch dimension
    image_tensor = torch.tensor(
        image_array
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        prediction = model(
            image_tensor
        )

        prediction = torch.sigmoid(
            prediction
        )

    # Convert probabilities to binary mask
    mask = (
        prediction[0, 0]
        .cpu()
        .numpy()
        > 0.5
    )

    return mask


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(mask, output_path):

    mask_image = (
        mask.astype(np.uint8)
        * 255
    )

    image = Image.fromarray(
        mask_image
    )

    image.save(output_path)

    print(
        f"Prediction saved to:\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Find project root
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # Trained model
    model_path = (
        project_root
        / "models"
        / "building_detection"
        / "building_model.pth"
    )

    # Test image
    image_path = (
        project_root
        / "data"
        / "processed"
        / "synthetic_buildings"
        / "images"
        / "building_001.png"
    )

    # Output folder
    output_folder = (
        project_root
        / "outputs"
        / "predictions"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / "building_001_prediction.png"
    )

    print("=" * 60)
    print("BUILDING DETECTION")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    print("\nLoading model...")

    model = load_model(
        model_path
    )

    print("Model loaded successfully!")

    print("\nRunning prediction...")

    mask = detect_buildings(
        image_path,
        model_path
    )

    print("Prediction completed!")

    print(
        "\nBuilding pixels detected:",
        int(mask.sum())
    )

    save_mask(
        mask,
        output_path
    )

    print("\nBuilding detection successful!")