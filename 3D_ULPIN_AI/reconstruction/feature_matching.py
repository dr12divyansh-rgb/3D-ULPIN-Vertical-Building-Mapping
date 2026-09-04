from pathlib import Path
import cv2
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUILDING_DIR = PROJECT_ROOT / "data" / "raw" / "buildings" / "building01"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def load_images():

    image_paths = sorted([
        path
        for path in BUILDING_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    images = []

    for path in image_paths:

        image = cv2.imread(str(path))

        if image is None:
            print(f"WARNING: Could not read {path.name}")
            continue

        images.append({
            "name": path.name,
            "path": path,
            "image": image
        })

    return images


def create_detector():

    # SIFT is useful for matching architectural features
    # such as windows, balconies and edges.

    return cv2.SIFT_create(
        nfeatures=3000
    )


def match_images(image_a, image_b, detector):

    gray_a = cv2.cvtColor(
        image_a,
        cv2.COLOR_BGR2GRAY
    )

    gray_b = cv2.cvtColor(
        image_b,
        cv2.COLOR_BGR2GRAY
    )

    keypoints_a, descriptors_a = detector.detectAndCompute(
        gray_a,
        None
    )

    keypoints_b, descriptors_b = detector.detectAndCompute(
        gray_b,
        None
    )

    if descriptors_a is None or descriptors_b is None:
        return 0, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2
    )

    # Lowe's ratio test
    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return (
        len(keypoints_a),
        len(good_matches)
    )


def main():

    print("=" * 60)
    print("3D ULPIN MULTI-VIEW FEATURE MATCHING")
    print("=" * 60)

    print(f"\nBuilding:")
    print(BUILDING_DIR)

    if not BUILDING_DIR.exists():

        print("\nERROR: Building folder not found.")
        return

    images = load_images()

    print(
        f"\nImages loaded: {len(images)}"
    )

    detector = create_detector()

    results = []

    print("\nMatching image pairs...\n")

    for i in range(len(images)):

        for j in range(i + 1, len(images)):

            image_a = images[i]
            image_b = images[j]

            keypoints, good_matches = match_images(
                image_a["image"],
                image_b["image"],
                detector
            )

            result = {
                "image_a": image_a["name"],
                "image_b": image_b["name"],
                "keypoints_a": keypoints,
                "good_matches": good_matches
            }

            results.append(result)

            print(
                f"{image_a['name'][:25]:25} ↔ "
                f"{image_b['name'][:25]:25} "
                f"| matches: {good_matches}"
            )

    results.sort(
        key=lambda x: x["good_matches"],
        reverse=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR /
        "building01_feature_matches.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("TOP IMAGE PAIRS")
    print("=" * 60)

    for result in results[:15]:

        print(
            f"{result['image_a']} ↔ "
            f"{result['image_b']} "
            f"→ {result['good_matches']} matches"
        )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()