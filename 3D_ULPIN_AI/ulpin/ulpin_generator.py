from pathlib import Path
import json
import hashlib


# ============================================================
# CONFIGURATION
# ============================================================

STATE_CODE = "TN"
DISTRICT_CODE = "VEL"


# ============================================================
# CREATE STABLE HASH
# ============================================================

def create_property_hash(property_data):
    """
    Create a stable hash from the property's
    important geometric and structural information.

    This is used only for prototype uniqueness.
    """

    geometry = property_data.get(
        "geometry",
        {}
    )

    polygon = geometry.get(
        "polygon",
        []
    )

    measurements = property_data.get(
        "measurements",
        {}
    )

    raw_data = {
        "polygon": polygon,
        "area": measurements.get(
            "area_pixels",
            0
        ),
        "height": measurements.get(
            "height_meters",
            0
        ),
        "floors": measurements.get(
            "estimated_floors",
            0
        )
    }

    raw_string = json.dumps(
        raw_data,
        sort_keys=True
    )

    hash_value = hashlib.sha256(
        raw_string.encode("utf-8")
    ).hexdigest()

    return hash_value


# ============================================================
# GENERATE ULPIN
# ============================================================

def generate_ulpin(property_data):
    """
    Generate a prototype ULPIN-style identifier.

    Format:

        TN-VEL-B001-XXXXXXXX

    The final official identifier format should
    be replaced when the government specification
    is available.
    """

    building_id = property_data[
        "building_id"
    ]

    property_hash = create_property_hash(
        property_data
    )

    short_hash = property_hash[
        :8
    ].upper()

    ulpin = (
        f"{STATE_CODE}-"
        f"{DISTRICT_CODE}-"
        f"{building_id}-"
        f"{short_hash}"
    )

    return ulpin


# ============================================================
# ADD ULPIN TO PROPERTY
# ============================================================

def assign_ulpin(property_data):
    """
    Generate and attach a ULPIN to a property.
    """

    ulpin = generate_ulpin(
        property_data
    )

    property_data["ulpin"] = ulpin

    return property_data


# ============================================================
# PROCESS ALL PROPERTIES
# ============================================================

def generate_all_ulpins(properties):

    results = []

    for property_data in properties:

        property_data = assign_ulpin(
            property_data
        )

        results.append(
            property_data
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

    input_path = (
        project_root
        / "outputs"
        / "property_data.json"
    )

    output_path = (
        project_root
        / "outputs"
        / "ulpin_properties.json"
    )

    print("=" * 60)
    print("ULPIN GENERATION")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # LOAD PROPERTY DATA
        # ----------------------------------------------------

        print(
            "\nLoading property data..."
        )

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as file:

            properties = json.load(
                file
            )

        print(
            "Properties loaded:",
            len(properties)
        )

        # ----------------------------------------------------
        # GENERATE ULPINS
        # ----------------------------------------------------

        print(
            "\nGenerating ULPIN identifiers..."
        )

        properties = generate_all_ulpins(
            properties
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        for property_data in properties:

            print(
                f"\nBuilding: "
                f"{property_data['building_id']}"
            )

            print(
                "ULPIN:",
                property_data["ulpin"]
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
                properties,
                file,
                indent=4
            )

        print(
            "\nULPIN data saved to:"
        )

        print(output_path)

        print(
            "\nULPIN generation successful!"
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )