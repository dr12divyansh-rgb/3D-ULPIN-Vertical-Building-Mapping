from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Ignore very small regions.
# This helps remove tiny noise from AI predictions.
MIN_BUILDING_AREA = 50


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):
    """
    Load a binary building mask.

    White pixels  = building
    Black pixels  = background
    """

    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    mask = Image.open(mask_path).convert("L")

    mask = np.array(mask)

    # Make sure mask is binary
    binary_mask = np.where(
        mask > 127,
        255,
        0
    ).astype(np.uint8)

    return binary_mask


# ============================================================
# EXTRACT POLYGONS
# ============================================================

def extract_building_polygons(mask):
    """
    Extract building polygons from a binary mask.

    Coordinates are currently pixel coordinates.
    """

    # Find connected building regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    buildings = []

    building_number = 1

    for contour in contours:

        # Calculate area
        area = cv2.contourArea(contour)

        # Ignore tiny regions
        if area < MIN_BUILDING_AREA:
            continue

        # Simplify contour
        perimeter = cv2.arcLength(
            contour,
            True
        )

        epsilon = 0.02 * perimeter

        simplified_contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        # Convert contour to normal Python coordinates
        polygon = []

        for point in simplified_contour:

            x, y = point[0]

            polygon.append([
                int(x),
                int(y)
            ])

        # Make building object
        building = {
            "building_id": f"B{building_number:03d}",
            "area_pixels": round(float(area), 2),
            "polygon": polygon
        }

        buildings.append(building)

        building_number += 1

    return buildings


# ============================================================
# SAVE BUILDINGS
# ============================================================

def save_buildings(buildings, output_path):
    """
    Save extracted building polygons as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            buildings,
            file,
            indent=4
        )

    print(
        f"\nBuilding data saved to:\n{output_path}"
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

    # AI prediction mask
    mask_path = (
        project_root
        / "outputs"
        / "predictions"
        / "building_001_prediction.png"
    )

    # Output file
    output_path = (
        project_root
        / "outputs"
        / "buildings.json"
    )

    print("=" * 60)
    print("BUILDING POLYGON EXTRACTION")
    print("=" * 60)

    try:

        # Load mask
        print("\nLoading building mask...")

        mask = load_mask(mask_path)

        print("Mask loaded successfully.")

        print(
            "Mask size:",
            mask.shape[1],
            "x",
            mask.shape[0]
        )

        # Extract polygons
        print("\nExtracting building polygons...")

        buildings = extract_building_polygons(
            mask
        )

        print(
            "Buildings detected:",
            len(buildings)
        )

        # Display information
        for building in buildings:

            print(
                f"\n{building['building_id']}"
            )

            print(
                "Area:",
                building["area_pixels"],
                "pixels"
            )

            print(
                "Polygon points:",
                len(building["polygon"])
            )

            print(
                "Coordinates:",
                building["polygon"]
            )

        # Save
        save_buildings(
            buildings,
            output_path
        )

        print(
            "\nPolygon extraction successful!"
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )