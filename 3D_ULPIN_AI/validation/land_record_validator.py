from pathlib import Path
import json

from shapely.geometry import Polygon


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum intersection-over-union ratio considered
# a reasonable geometric match.
MATCH_THRESHOLD = 0.70

# Below this, the building is considered largely
# outside the parcel.
OUTSIDE_THRESHOLD = 0.10


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path):

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CREATE POLYGON
# ============================================================

def create_polygon(coordinates):

    polygon = Polygon(
        coordinates
    )

    # Fix small geometry problems
    if not polygon.is_valid:

        polygon = polygon.buffer(0)

    return polygon


# ============================================================
# CALCULATE IOU
# ============================================================

def calculate_iou(
    building_polygon,
    parcel_polygon
):
    """
    Intersection over Union:

        IoU = intersection / union

    Higher value = stronger geometric overlap.
    """

    intersection = (
        building_polygon
        .intersection(parcel_polygon)
        .area
    )

    union = (
        building_polygon
        .union(parcel_polygon)
        .area
    )

    if union == 0:

        return 0.0

    return intersection / union


# ============================================================
# BUILDING INSIDE PARCEL RATIO
# ============================================================

def calculate_building_coverage(
    building_polygon,
    parcel_polygon
):
    """
    Calculates what percentage of the building
    lies inside the parcel.
    """

    building_area = (
        building_polygon.area
    )

    if building_area == 0:

        return 0.0

    intersection = (
        building_polygon
        .intersection(parcel_polygon)
        .area
    )

    return (
        intersection
        / building_area
    )


# ============================================================
# VALIDATE BUILDING
# ============================================================

def validate_building(
    building,
    parcels
):
    """
    Find the best matching land parcel
    for a detected building.
    """

    building_polygon = create_polygon(
        building["polygon"]
    )

    best_match = None

    best_iou = 0.0

    best_coverage = 0.0

    for parcel in parcels:

        parcel_polygon = create_polygon(
            parcel["polygon"]
        )

        iou = calculate_iou(
            building_polygon,
            parcel_polygon
        )

        coverage = calculate_building_coverage(
            building_polygon,
            parcel_polygon
        )

        if iou > best_iou:

            best_iou = iou

            best_coverage = coverage

            best_match = parcel

    # No meaningful overlap
    if best_match is None:

        return {
            "building_id": building["building_id"],
            "validation_status": "UNREGISTERED",
            "parcel_id": None,
            "overlap_iou": 0.0,
            "building_coverage": 0.0
        }

    # Building almost completely outside parcel
    if best_coverage < OUTSIDE_THRESHOLD:

        status = "OUTSIDE_PARCEL"

    # Strong geometric match
    elif best_iou >= MATCH_THRESHOLD:

        status = "MATCH"

    # Some overlap but not a strong match
    else:

        status = "PARTIAL_MATCH"

    return {
        "building_id": building["building_id"],
        "validation_status": status,
        "parcel_id": best_match["parcel_id"],
        "overlap_iou": round(
            best_iou,
            4
        ),
        "building_coverage": round(
            best_coverage,
            4
        )
    }


# ============================================================
# VALIDATE ALL BUILDINGS
# ============================================================

def validate_all_buildings(
    buildings,
    parcels
):

    results = []

    for building in buildings:

        result = validate_building(
            building,
            parcels
        )

        results.append(
            result
        )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    buildings_path = (
        project_root
        / "outputs"
        / "buildings.json"
    )

    parcels_path = (
        project_root
        / "data"
        / "mock_land_records.json"
    )

    output_path = (
        project_root
        / "outputs"
        / "validation_results.json"
    )

    print("=" * 60)
    print("LAND RECORD VALIDATION")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        print("\nLoading detected buildings...")

        buildings = load_json(
            buildings_path
        )

        print(
            "Buildings loaded:",
            len(buildings)
        )

        print("\nLoading land records...")

        parcels = load_json(
            parcels_path
        )

        print(
            "Parcels loaded:",
            len(parcels)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print(
            "\nComparing building footprints "
            "with land parcels..."
        )

        results = validate_all_buildings(
            buildings,
            parcels
        )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        print("\n" + "-" * 60)

        for result in results:

            print(
                f"\nBuilding: "
                f"{result['building_id']}"
            )

            print(
                "Status:",
                result["validation_status"]
            )

            print(
                "Parcel:",
                result["parcel_id"]
            )

            print(
                "IoU:",
                result["overlap_iou"]
            )

            print(
                "Building inside parcel:",
                f"{result['building_coverage'] * 100:.2f}%"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

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
                results,
                file,
                indent=4
            )

        print(
            "\nValidation results saved to:"
        )

        print(output_path)

        print(
            "\nLand record validation successful!"
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )