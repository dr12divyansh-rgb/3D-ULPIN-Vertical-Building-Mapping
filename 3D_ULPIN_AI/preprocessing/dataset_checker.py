from pathlib import Path


# Supported image formats
IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}

# Supported annotation formats
ANNOTATION_FORMATS = {
    ".json",
    ".geojson",
    ".shp",
    ".xml",
    ".txt"
}


def find_files(folder, extensions):
    """
    Find files with specific extensions.
    """

    folder = Path(folder)

    files = []

    if not folder.exists():
        return files

    for file in folder.rglob("*"):

        if file.is_file() and file.suffix.lower() in extensions:
            files.append(file)

    return files


def check_dataset_folder(folder):
    """
    Check the contents of a dataset folder.
    """

    folder = Path(folder)

    print("=" * 50)
    print("DATASET CHECK")
    print("=" * 50)

    print(f"\nDataset folder:")
    print(folder)

    if not folder.exists():

        print("\nFolder does not exist.")
        return

    # Find images
    images = find_files(
        folder,
        IMAGE_FORMATS
    )

    # Find annotations
    annotations = find_files(
        folder,
        ANNOTATION_FORMATS
    )

    print("\nImages found:", len(images))
    print("Annotations found:", len(annotations))

    # Display images
    if images:

        print("\nImage files:")

        for image in images:
            print("  -", image.name)

    else:

        print("\nNo image files found.")

    # Display annotations
    if annotations:

        print("\nAnnotation files:")

        for annotation in annotations:
            print("  -", annotation.name)

    else:

        print("\nNo annotation files found.")

    print("\n" + "=" * 50)


if __name__ == "__main__":

    # Project root
    project_root = Path(__file__).resolve().parent.parent

    # Dataset location
    dataset_folder = (
        project_root
        / "data"
        / "raw"
    )

    check_dataset_folder(dataset_folder)