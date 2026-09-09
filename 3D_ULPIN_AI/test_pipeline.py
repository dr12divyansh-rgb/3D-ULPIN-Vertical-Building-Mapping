from pathlib import Path
import json
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT MODULES
# ============================================================

from geospatial.polygon_processing import (
    load_mask,
    extract_building_polygons
)

from integration.property_builder import (
    load_buildings,
    build_property_dataset
)

from validation.land_record_validator import (
    load_json,
    validate_all_buildings
)

from ulpin.ulpin_generator import (
    generate_all_ulpins
)


# ============================================================
# PATHS
# ============================================================

MASK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "building_001_prediction.png"
)

LAND_RECORD_PATH = (
    PROJECT_ROOT
    / "data"
    / "mock_land_records.json"
)

BUILDINGS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "buildings.json"
)

PROPERTY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "property_data.json"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation_results.json"
)

FINAL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_property_data.json"
)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# STEP 1 — BUILDING POLYGONS
# ============================================================

def run_building_extraction():

    print("\n[1/4] BUILDING EXTRACTION")
    print("-" * 60)

    mask = load_mask(
        MASK_PATH
    )

    buildings = extract_building_polygons(
        mask
    )

    print(
        "Buildings detected:",
        len(buildings)
    )

    save_json(
        buildings,
        BUILDINGS_PATH
    )

    return buildings


# ============================================================
# STEP 2 — PROPERTY DATA
# ============================================================

def run_property_builder(buildings):

    print("\n[2/4] PROPERTY DATA")
    print("-" * 60)

    properties = build_property_dataset(
        buildings
    )

    save_json(
        properties,
        PROPERTY_PATH
    )

    print(
        "Properties created:",
        len(properties)
    )

    return properties


# ============================================================
# STEP 3 — LAND RECORD VALIDATION
# ============================================================

def run_validation(buildings):

    print("\n[3/4] LAND RECORD VALIDATION")
    print("-" * 60)

    parcels = load_json(
        LAND_RECORD_PATH
    )

    results = validate_all_buildings(
        buildings,
        parcels
    )

    save_json(
        results,
        VALIDATION_PATH
    )

    for result in results:

        print(
            result["building_id"],
            "→",
            result["validation_status"]
        )

    return results


# ============================================================
# STEP 4 — ULPIN GENERATION
# ============================================================

def run_ulpin_generation(properties):

    print("\n[4/4] ULPIN GENERATION")
    print("-" * 60)

    properties = generate_all_ulpins(
        properties
    )

    save_json(
        properties,
        FINAL_PATH
    )

    for property_data in properties:

        print(
            property_data["building_id"],
            "→",
            property_data["building_property_id"]
        )

    return properties


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("3D ULPIN AI / ML MASTER PIPELINE")
    print("=" * 70)

    print(
        "\nProject:",
        PROJECT_ROOT
    )

    try:

        # ----------------------------------------------------
        # 1. BUILDING EXTRACTION
        # ----------------------------------------------------

        buildings = run_building_extraction()

        # ----------------------------------------------------
        # 2. PROPERTY DATA
        # ----------------------------------------------------

        properties = run_property_builder(
            buildings
        )

        # ----------------------------------------------------
        # 3. VALIDATION
        # ----------------------------------------------------

        validation_results = run_validation(
            buildings
        )

        # ----------------------------------------------------
        # MERGE VALIDATION RESULTS INTO PROPERTIES
        # ----------------------------------------------------

        for property_data in properties:

            building_id = property_data["building_id"]

            matching_result = next(
                (
                    result
                    for result in validation_results
                    if result["building_id"] == building_id
                ),
                None
            )

            if matching_result:

                property_data["validation"] = {

                    "status": matching_result[
                        "validation_status"
                    ],

                    "parcel_id": matching_result[
                        "parcel_id"
                    ],

                    "overlap_iou": matching_result[
                        "overlap_iou"
                    ],

                    "building_coverage": matching_result[
                        "building_coverage"
                        ]
                }

        # ----------------------------------------------------
        # 4. ULPIN
        # ----------------------------------------------------

        final_properties = run_ulpin_generation(
            properties
        )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        print(
            "\nFinal property records:",
            len(final_properties)
        )

        print(
            "\nFinal output:"
        )

        print(
            FINAL_PATH
        )

        print(
            "\nAll stages completed successfully!"
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)

        print(
            "\nERROR:",
            error
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()