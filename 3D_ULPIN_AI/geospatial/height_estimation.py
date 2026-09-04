from pathlib import Path
import json

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Prototype average floor height
AVERAGE_FLOOR_HEIGHT = 3.0


# ============================================================
# HEIGHT CALCULATION
# ============================================================

def calculate_building_height(
    dsm_values,
    dem_values
):
    """
    Calculate building height using:

        Height = DSM - DEM

    dsm_values:
        Elevation values including buildings.

    dem_values:
        Ground elevation values.
    """

    dsm_values = np.array(
        dsm_values,
        dtype=float
    )

    dem_values = np.array(
        dem_values,
        dtype=float
    )

    height_values = (
        dsm_values
        - dem_values
    )

    # Remove negative values
    height_values = np.maximum(
        height_values,
        0
    )

    return height_values


# ============================================================
# REPRESENTATIVE BUILDING HEIGHT
# ============================================================

def estimate_building_height(
    dsm_values,
    dem_values
):
    """
    Estimate a representative building height.

    We use the median instead of the maximum
    because maximum values may contain noise.
    """

    heights = calculate_building_height(
        dsm_values,
        dem_values
    )

    if len(heights) == 0:

        return 0.0

    height = np.median(
        heights
    )

    return float(height)


# ============================================================
# FLOOR ESTIMATION
# ============================================================

def estimate_floors(
    height,
    floor_height=AVERAGE_FLOOR_HEIGHT
):
    """
    Estimate number of floors from building height.
    """

    if height <= 0:

        return 0

    floors = round(
        height / floor_height
    )

    return max(
        floors,
        1
    )


# ============================================================
# BUILDING ANALYSIS
# ============================================================

def analyze_building(
    building_id,
    dsm_values,
    dem_values
):
    """
    Calculate height and estimated floors
    for one building.
    """

    height = estimate_building_height(
        dsm_values,
        dem_values
    )

    floors = estimate_floors(
        height
    )

    return {
        "building_id": building_id,
        "height_meters": round(
            height,
            2
        ),
        "estimated_floors": floors
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING HEIGHT ESTIMATION")
    print("=" * 60)

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------
    #
    # These values represent elevation samples
    # over a hypothetical building.
    #

    dsm_values = [
        143.5,
        143.8,
        144.0,
        143.7,
        143.9,
        143.6
    ]

    dem_values = [
        128.5,
        128.7,
        128.6,
        128.8,
        128.5,
        128.7
    ]

    # Analyze building
    result = analyze_building(
        building_id="B001",
        dsm_values=dsm_values,
        dem_values=dem_values
    )

    print("\nBuilding ID:")
    print(result["building_id"])

    print("\nEstimated height:")
    print(
        result["height_meters"],
        "meters"
    )

    print("\nEstimated floors:")
    print(
        result["estimated_floors"]
    )

    print("\nHeight estimation successful!")