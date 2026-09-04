from pathlib import Path
import json


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_property_data.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "frontend_properties.json"
)

JS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "frontend_properties.js"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_properties():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CALCULATE BOUNDING BOX
# ============================================================

def calculate_bbox(polygon):

    if not polygon:
        return 0, 0, 0, 0

    xs = [
        point[0]
        for point in polygon
    ]

    ys = [
        point[1]
        for point in polygon
    ]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    width = max_x - min_x
    depth = max_y - min_y

    center_x = (
        min_x + max_x
    ) / 2

    center_y = (
        min_y + max_y
    ) / 2

    return (
        center_x,
        center_y,
        width,
        depth
    )


# ============================================================
# CONVERT PROPERTY
# ============================================================

def convert_property(
    property_data,
    index
):
    """
    Convert backend property data into a format
    that the Three.js frontend can understand.

    Current coordinates are synthetic pixel coordinates.

    Later, these will be replaced with real geographic
    coordinates from GeoTIFF / GIS data.
    """

    building_id = property_data[
        "building_id"
    ]

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

    validation = property_data.get(
        "validation",
        {}
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    center_x, center_y, width, depth = (
        calculate_bbox(polygon)
    )

    # --------------------------------------------------------
    # Scale pixel coordinates to the
    # current Three.js city coordinate system.
    #
    # This is ONLY for visualization.
    # --------------------------------------------------------

    SCALE = 0.35

    x = (
        center_x
        - 64
    ) * SCALE

    z = (
        center_y
        - 64
    ) * SCALE

    width = max(
        width * SCALE,
        2
    )

    depth = max(
        depth * SCALE,
        2
    )

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    height = measurements.get(
        "height_meters",
        6
    )

    floors = measurements.get(
        "estimated_floors",
        max(
            1,
            round(height / 3)
        )
    )

    # --------------------------------------------------------
    # Determine type
    #
    # Our current synthetic AI dataset does not contain
    # land-use classification, so we use a temporary
    # deterministic assignment.
    #
    # This will later come from land records / GIS.
    # --------------------------------------------------------

    types = [
        "residential",
        "commercial",
        "institutional"
    ]

    property_type = types[
        index % len(types)
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_status = validation.get(
        "status",
        "NOT_VALIDATED"
    )

    # --------------------------------------------------------
    # Frontend object
    # --------------------------------------------------------

    frontend_property = {

        "id": index,

        "building_id": building_id,

        "x": round(
            x,
            3
        ),

        "z": round(
            z,
            3
        ),

        "w": round(
            width,
            3
        ),

        "d": round(
            depth,
            3
        ),

        "h": round(
            height,
            3
        ),

        "floors": int(
            floors
        ),

        "type": property_type,

        "hasBasement": False,

        "ulpin": property_data.get(
            "ulpin"
        ),

        "area_pixels": measurements.get(
            "area_pixels",
            0
        ),

        "polygon": polygon,

        "validation": {

            "status": validation_status,

            "parcel_id": validation.get(
                "parcel_id"
            ),

            "overlap_iou": validation.get(
                "overlap_iou",
                0
            ),

            "building_coverage": validation.get(
                "building_coverage",
                0
            )
        }
    }

    return frontend_property


# ============================================================
# BUILD FRONTEND DATASET
# ============================================================

def build_frontend_dataset(properties):

    frontend_properties = []

    for index, property_data in enumerate(
        properties
    ):

        converted = convert_property(
            property_data,
            index
        )

        frontend_properties.append(
            converted
        )

    return frontend_properties


# ============================================================
# SAVE
# ============================================================

def save_dataset(data):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

# ============================================================
# SAVE JAVASCRIPT DATA
# ============================================================

def save_javascript_dataset(data):

    JS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        JS_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "window.ULPIN_PROPERTIES = "
        )

        json.dump(
            data,
            file,
            indent=4
        )

        file.write(";")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FRONTEND DATA ADAPTER")
    print("=" * 60)

    try:

        print(
            "\nLoading final property data..."
        )

        properties = load_properties()

        print(
            "Properties loaded:",
            len(properties)
        )

        print(
            "\nConverting properties..."
        )

        frontend_data = (
            build_frontend_dataset(
                properties
            )
        )

        save_dataset(
            frontend_data
        )

        save_javascript_dataset(
            frontend_data
        )

        print(
            "\nFrontend properties generated:",
            len(frontend_data)
        )

        print(
            "\nJSON saved to:"
        )

        print(
            OUTPUT_PATH
        )

        print(
            "\nJavaScript data saved to:"
        )

        print(
            JS_OUTPUT_PATH
        )

        print(
            "\nFrontend data adapter successful!"
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )