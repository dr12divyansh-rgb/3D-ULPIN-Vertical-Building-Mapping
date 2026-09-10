from pathlib import Path
import json
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORT MODULES
# ============================================================

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

LAND_RECORD_PATH = (
    PROJECT_ROOT
    / "data"
    / "mock_land_records.json"
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("3D ULPIN - 20 BUILDING MASTER PIPELINE")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # STEP 1 — LOAD 20 BUILDINGS
        # ----------------------------------------------------

        print("\n[1/4] LOADING BUILDING DATA")
        print("-" * 60)

        buildings = load_buildings(
            BUILDINGS_PATH
        )

        print(
            "Buildings loaded:",
            len(buildings)
        )

        if len(buildings) != 20:

            print(
                "\nWARNING: Expected 20 buildings."
            )

            print(
                "Current buildings:",
                len(buildings)
            )

        # ----------------------------------------------------
        # STEP 2 — PROPERTY DATA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # STEP 3 — LAND RECORD VALIDATION
        # ----------------------------------------------------

        print("\n[3/4] LAND RECORD VALIDATION")
        print("-" * 60)

        parcels = load_json(
            LAND_RECORD_PATH
        )

        validation_results = validate_all_buildings(
            buildings,
            parcels
        )

        save_json(
            validation_results,
            VALIDATION_PATH
        )

        print(
            "Validation records:",
            len(validation_results)
        )

        for result in validation_results:

            print(
                result["building_id"],
                "->",
                result["validation_status"]
            )

        # ----------------------------------------------------
        # MERGE VALIDATION INTO PROPERTY DATA
        # ----------------------------------------------------

        for property_data in properties:

            building_id = property_data[
                "building_id"
            ]

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

                    "parcel_id": matching_result.get(
                        "parcel_id"
                    ),

                    "overlap_iou": matching_result.get(
                        "overlap_iou"
                    ),

                    "building_coverage": matching_result.get(
                        "building_coverage"
                    )
                }

        # ----------------------------------------------------
        # STEP 4 — 3D PROPERTY IDs
        # ----------------------------------------------------

        print("\n[4/4] 3D PROPERTY ID GENERATION")
        print("-" * 60)

        final_properties = generate_all_ulpins(
            properties
        )

        save_json(
            final_properties,
            FINAL_PATH
        )

        for property_data in final_properties:

            print(
                property_data["building_id"],
                "->",
                property_data[
                    "building_property_id"
                ]
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("20-BUILDING PIPELINE COMPLETE")
        print("=" * 70)

        print(
            "\nBuildings:",
            len(buildings)
        )

        print(
            "Properties:",
            len(properties)
        )

        print(
            "Final records:",
            len(final_properties)
        )

        print(
            "\nFinal output:"
        )

        print(
            FINAL_PATH
        )

        print(
            "\nFrontend can now use these records."
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