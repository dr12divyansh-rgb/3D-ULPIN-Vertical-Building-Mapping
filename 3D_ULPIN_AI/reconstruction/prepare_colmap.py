from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "buildings"
)

COLMAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "reconstruction"
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def prepare_building(building_dir):

    building_id = building_dir.name

    output_dir = (
        COLMAP_ROOT
        / building_id
    )

    image_dir = (
        output_dir
        / "images"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted([
        file
        for file in building_dir.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ])

    copied = 0

    for image_file in image_files:

        destination = (
            image_dir
            / image_file.name
        )

        if not destination.exists():

            shutil.copy2(
                image_file,
                destination
            )

        copied += 1

    return {
        "building_id": building_id,
        "source_images": len(image_files),
        "prepared_images": copied,
        "image_directory": str(image_dir)
    }


def main():

    print("=" * 60)
    print("3D ULPIN COLMAP DATASET PREPARATION")
    print("=" * 60)

    if not SOURCE_ROOT.exists():

        print("\nERROR:")
        print("Source building dataset not found:")
        print(SOURCE_ROOT)
        return

    building_dirs = sorted([
        directory
        for directory in SOURCE_ROOT.iterdir()
        if directory.is_dir()
    ])

    print(
        f"\nBuildings found: {len(building_dirs)}"
    )

    total_images = 0

    for building_dir in building_dirs:

        result = prepare_building(
            building_dir
        )

        total_images += result["prepared_images"]

        print(
            f"{result['building_id']}: "
            f"{result['prepared_images']} images prepared"
        )

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal buildings: "
        f"{len(building_dirs)}"
    )

    print(
        f"Total images: "
        f"{total_images}"
    )

    print(
        "\nReconstruction workspace:"
    )

    print(COLMAP_ROOT)


if __name__ == "__main__":
    main()