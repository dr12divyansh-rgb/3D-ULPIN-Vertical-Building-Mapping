from pathlib import Path
import json


# ============================================================
# PROTOTYPE GOVERNMENT ULPIN
# ============================================================
# In the real system, this will come from government
# land-record data.
#
# Official ULPIN is associated with the LAND PARCEL.
# These are only prototype placeholders for now.
# ============================================================

PROTOTYPE_ULPIN_BY_PARCEL = {
    "P001": "1234567890ABCD",
    "P002": "2345678901BCDE",
    "P003": "3456789012CDEF",
}


# ============================================================
# GET BASE / GOVERNMENT ULPIN
# ============================================================

def get_official_ulpin(parcel_id):

    return PROTOTYPE_ULPIN_BY_PARCEL.get(
        parcel_id,
        "00000000000000"
    )


# ============================================================
# GENERATE 3D PROPERTY ID
# ============================================================

def generate_3d_property_id(
    official_ulpin,
    building_id,
    floor_number,
    flat_id
):

    floor_id = f"F{int(floor_number):02d}"

    flat_id = str(flat_id)

    # Remove existing U prefix if present
    if flat_id.startswith("U"):
        flat_id = flat_id[1:]

    return (
        f"{official_ulpin}-"
        f"{building_id}-"
        f"{floor_id}-"
        f"U{flat_id}"
    )


# ============================================================
# GENERATE BUILDING PROPERTY ID
# ============================================================

def generate_building_property_id(
    official_ulpin,
    building_id
):

    return f"{official_ulpin}-{building_id}"


# ============================================================
# MAIN ULPIN / PROPERTY ID FUNCTION
# ============================================================

def generate_all_ulpins(properties):

    for property_data in properties:

        building_id = property_data.get(
            "building_id",
            "B001"
        )

        # ----------------------------------------------------
        # LAND RECORD INFORMATION
        # ----------------------------------------------------

        validation = property_data.get(
            "validation",
            {}
        )

        parcel_id = validation.get(
            "parcel_id"
        )

        # ----------------------------------------------------
        # GOVERNMENT ULPIN
        # ----------------------------------------------------

        official_ulpin = get_official_ulpin(
            parcel_id
        )

        property_data["official_ulpin"] = (
            official_ulpin
        )

        property_data["ulpin_status"] = (
            "PROTOTYPE_PLACEHOLDER"
        )

        # ----------------------------------------------------
        # OUR BUILDING ID
        # ----------------------------------------------------

        property_data["building_id"] = (
            building_id
        )

        property_data["building_property_id"] = (
            generate_building_property_id(
                official_ulpin,
                building_id
            )
        )

        # ----------------------------------------------------
        # APARTMENT / FLOOR INFORMATION
        # ----------------------------------------------------

        apartments = property_data.get(
            "apartments",
            {}
        )

        floors = apartments.get(
            "floors",
            []
        )

        for floor in floors:

            floor_number = floor.get(
                "floor_number",
                0
            )

            # Our internal floor ID
            floor["floor_id"] = (
                f"F{int(floor_number):02d}"
            )

            flat_ids = floor.get(
                "flats",
                []
            )

            floor["3d_property_ids"] = []

            # ------------------------------------------------
            # CREATE NEW ID FOR EACH FLAT
            # ------------------------------------------------

            for flat_id in flat_ids:

                new_property_id = (
                    generate_3d_property_id(
                        official_ulpin,
                        building_id,
                        floor_number,
                        flat_id
                    )
                )

                floor[
                    "3d_property_ids"
                ].append(
                    new_property_id
                )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        property_data[
            "identifier_status"
        ] = "PROTOTYPE"

    return properties