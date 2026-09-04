from pathlib import Path
from PIL import Image
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUILDINGS_DIR = PROJECT_ROOT / "data" / "raw" / "buildings"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def inspect_image(image_path):
    try:
        with Image.open(image_path) as img:
            return {
                "filename": image_path.name,
                "extension": image_path.suffix.lower(),
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "size_mb": round(image_path.stat().st_size / (1024 * 1024), 3)
            }

    except Exception as error:
        return {
            "filename": image_path.name,
            "error": str(error)
        }


def classify_view(filename):
    name = filename.lower()

    if "front" in name:
        return "front"

    if "left" in name:
        return "left"

    if "right" in name:
        return "right"

    if "low" in name:
        return "low"

    if "zoom" in name:
        return "zoom"

    return "unknown"


def inspect_building(building_dir):
    image_files = [
        file for file in building_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    images = []

    for image_file in sorted(image_files):
        image_info = inspect_image(image_file)
        image_info["view_type"] = classify_view(image_file.name)
        images.append(image_info)

    return {
        "building_id": building_dir.name,
        "image_count": len(images),
        "images": images
    }


def main():

    if not BUILDINGS_DIR.exists():
        print(f"ERROR: Dataset folder not found:")
        print(BUILDINGS_DIR)
        return

    building_dirs = sorted([
        directory
        for directory in BUILDINGS_DIR.iterdir()
        if directory.is_dir()
    ])

    print("=" * 60)
    print("3D ULPIN DATASET INVENTORY")
    print("=" * 60)

    print(f"\nDataset location:")
    print(BUILDINGS_DIR)

    print(f"\nBuildings found: {len(building_dirs)}")

    dataset = []

    for building_dir in building_dirs:

        result = inspect_building(building_dir)

        dataset.append(result)

        print(
            f"{result['building_id']}: "
            f"{result['image_count']} images"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "dataset_inventory.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4)

    print("\n" + "=" * 60)
    print("VIEW DISTRIBUTION")
    print("=" * 60)

    view_counts = {}

    for building in dataset:
        for image in building["images"]:

            view = image.get("view_type", "unknown")

            view_counts[view] = view_counts.get(view, 0) + 1

    for view, count in sorted(view_counts.items()):
        print(f"{view:10} : {count}")

    print("\nInventory saved to:")
    print(output_file)


if __name__ == "__main__":
    main()