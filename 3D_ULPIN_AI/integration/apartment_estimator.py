# ============================================================
# APARTMENT / FLAT ESTIMATOR
# ============================================================

"""
Prototype apartment estimation module.

This module estimates the number of apartments/flats on each
floor using the available building area.

IMPORTANT:
These values are ESTIMATES for the prototype.

They are NOT extracted from actual floor plans or government
property records.

Later, this module can be replaced with actual:
    - floor plans
    - municipal property records
    - GIS data
    - architectural data
    - AI-based unit detection
"""


# ============================================================
# CONFIGURATION
# ============================================================

# Approximate area assumed for one apartment.
# This is a prototype assumption and can be changed later.
AVERAGE_APARTMENT_AREA = 70.0

# Minimum number of apartments per floor.
MIN_FLATS_PER_FLOOR = 1

# Maximum number of apartments per floor.
MAX_FLATS_PER_FLOOR = 10


# ============================================================
# ESTIMATE FLATS PER FLOOR
# ============================================================

def estimate_flats_per_floor(
    floor_area
):
    """
    Estimate the number of apartments on one floor.

    Parameters
    ----------
    floor_area : float
        Estimated usable floor area.

    Returns
    -------
    int
        Estimated number of flats.
    """

    if floor_area <= 0:

        return MIN_FLATS_PER_FLOOR

    estimated_flats = round(
        floor_area / AVERAGE_APARTMENT_AREA
    )

    estimated_flats = max(
        MIN_FLATS_PER_FLOOR,
        estimated_flats
    )

    estimated_flats = min(
        MAX_FLATS_PER_FLOOR,
        estimated_flats
    )

    return estimated_flats


# ============================================================
# GENERATE FLAT IDs
# ============================================================

def generate_flat_ids(
    floor_number,
    flat_count
):
    """
    Generate prototype apartment IDs.

    Example:

        Floor 0:
            G01, G02, G03

        Floor 1:
            101, 102, 103

        Floor 2:
            201, 202, 203
    """

    flat_ids = []

    for index in range(
        1,
        flat_count + 1
    ):

        if floor_number == 0:

            flat_id = (
                f"G{index:02d}"
            )

        else:

            flat_id = (
                f"{floor_number}"
                f"{index:02d}"
            )

        flat_ids.append(
            flat_id
        )

    return flat_ids


# ============================================================
# ESTIMATE APARTMENTS FOR BUILDING
# ============================================================

def estimate_apartments(
    building_area,
    floors
):
    """
    Estimate apartment distribution across all floors.

    The prototype assumes that the building footprint represents
    the approximate floor area.

    Parameters
    ----------
    building_area : float
        Building area in pixel units.

    floors : int
        Estimated number of floors.

    Returns
    -------
    dict
        Apartment estimation information.
    """

    if floors <= 0:

        floors = 1

    # --------------------------------------------------------
    # Current prototype assumption:
    #
    # Building footprint area ≈ floor area.
    #
    # This is intentionally marked as an assumption.
    # --------------------------------------------------------

    floor_area = (
        building_area
    )

    flats_per_floor = (
        estimate_flats_per_floor(
            floor_area
        )
    )

    floor_data = []

    total_flats = 0

    for floor_number in range(
        floors
    ):

        flat_ids = generate_flat_ids(
            floor_number,
            flats_per_floor
        )

        floor_record = {

            "floor_number": floor_number,

            "floor_label": (
                "Ground Floor"
                if floor_number == 0
                else f"Floor {floor_number}"
            ),

            "estimated_floor_area": round(
                floor_area,
                2
            ),

            "estimated_flat_count": (
                flats_per_floor
            ),

            "flats": flat_ids,

            "data_source": (
                "PROTOTYPE_ESTIMATE"
            )
        }

        floor_data.append(
            floor_record
        )

        total_flats += (
            flats_per_floor
        )

    return {

        "total_estimated_flats": (
            total_flats
        ),

        "estimated_flats_per_floor": (
            flats_per_floor
        ),

        "average_apartment_area_assumption": (
            AVERAGE_APARTMENT_AREA
        ),

        "data_source": (
            "PROTOTYPE_ESTIMATE"
        ),

        "status": (
            "ESTIMATED"
        ),

        "floors": floor_data
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("APARTMENT ESTIMATOR TEST")
    print("=" * 60)

    test_area = 240
    test_floors = 5

    result = estimate_apartments(
        test_area,
        test_floors
    )

    print(
        "\nTotal estimated flats:",
        result[
            "total_estimated_flats"
        ]
    )

    for floor in result["floors"]:

        print(
            "\n",
            floor["floor_label"],
            "→",
            floor["estimated_flat_count"],
            "flats"
        )

        print(
            "Flats:",
            floor["flats"]
        )