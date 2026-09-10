from pathlib import Path
import json
import re

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

BUILDINGS_DIR = ROOT / "data" / "raw" / "buildings"
OUTPUT_DIR = ROOT / "outputs"

NUMBER_OF_BUILDINGS = 20


# ---------------------------------------------------------
# FIND BUILDING FOLDERS
# ---------------------------------------------------------

building_dirs = []

for folder in BUILDINGS_DIR.iterdir():
    if folder.is_dir() and re.match(r"building\d+", folder.name.lower()):
        building_dirs.append(folder)

building_dirs.sort(
    key=lambda p: int(re.search(r"\d+", p.name).group())
)

building_dirs = building_dirs[:NUMBER_OF_BUILDINGS]


if not building_dirs:
    raise RuntimeError(
        f"No building folders found in {BUILDINGS_DIR}"
    )


# ---------------------------------------------------------
# GENERATE BUILDINGS
# ---------------------------------------------------------

properties = []

# 5 x 4 city arrangement
columns = 5

for index, folder in enumerate(building_dirs):

    building_number = index + 1
    building_id = f"B{building_number:03d}"

    # -----------------------------------------------------
    # Use image count as a real dataset signal
    # -----------------------------------------------------

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    image_count = len(images)

    # Different deterministic geometry for each building
    width = 8 + ((building_number * 3) % 9)
    depth = 7 + ((building_number * 5) % 8)

    # Different heights / floors
    floors = 2 + ((building_number * 7) % 6)

    height = floors * 3.0

    # -----------------------------------------------------
    # Arrange buildings in a city-like grid
    # -----------------------------------------------------

    row = index // columns
    col = index % columns

    spacing_x = 22
    spacing_z = 20

    center_x = 18 + col * spacing_x
    center_z = 18 + row * spacing_z

    x1 = center_x - width / 2
    x2 = center_x + width / 2
    z1 = center_z - depth / 2
    z2 = center_z + depth / 2

    polygon = [
        [x1, z1],
        [x2, z1],
        [x2, z2],
        [x1, z2]
    ]

    area = width * depth

    # Different building types
    types = [
        "residential",
        "residential",
        "commercial",
        "residential",
        "institutional"
    ]

    building_type = types[index % len(types)]

    properties.append({
        "building_id": building_id,

        "source_folder": folder.name,
        "source_image_count": image_count,

        "polygon": polygon,
        "area_pixels": area,

        "x": center_x,
        "z": center_z,

        "w": width,
        "d": depth,
        "h": height,

        "floors": floors,

        "type": building_type,

        "hasBasement": (
            building_number % 5 == 0
        ),

        "detection_status": "DATASET_DERIVED_PROTOTYPE",
        "geometry_status": "PROTOTYPE_ESTIMATE"
    })


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "buildings.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(properties, f, indent=2)


# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

print("=" * 55)
print("MULTI-BUILDING GENERATOR")
print("=" * 55)

print(f"Dataset folder : {BUILDINGS_DIR}")
print(f"Buildings used : {len(properties)}")
print()

for p in properties:
    print(
        f"{p['building_id']}  "
        f"{p['source_folder']}  "
        f"{p['source_image_count']} images  "
        f"{p['floors']} floors  "
        f"{p['w']}x{p['d']} m"
    )

print()
print(f"Saved: {output_file}")
print("=" * 55)