from pathlib import Path
import cv2
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUILDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "buildings"
    / "building01"
)

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
            "image": image
        })

    return images


def get_good_matches(
    image_a,
    image_b,
    detector,
    ratio=0.75
):

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
        return [], keypoints_a, keypoints_b

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return (
        good_matches,
        keypoints_a,
        keypoints_b
    )


def verify_geometry(
    image_a,
    image_b,
    detector
):

    matches, keypoints_a, keypoints_b = get_good_matches(
        image_a,
        image_b,
        detector
    )

    if len(matches) < 8:

        return {
            "raw_matches": len(matches),
            "fundamental_inliers": 0,
            "fundamental_ratio": 0,
            "homography_inliers": 0,
            "homography_ratio": 0
        }

    points_a = []
    points_b = []

    for match in matches:

        points_a.append(
            keypoints_a[match.queryIdx].pt
        )

        points_b.append(
            keypoints_b[match.trainIdx].pt
        )

    points_a = []
    points_b = []

    for match in matches:
        points_a.append(keypoints_a[match.queryIdx].pt)
        points_b.append(keypoints_b[match.trainIdx].pt)

    points_a = np.array(
        points_a,
        dtype=np.float32
    )

    points_b = np.array(
        points_b,
        dtype=np.float32
    )

    # Fundamental matrix
    fundamental_inliers = 0

    try:

        F, mask_f = cv2.findFundamentalMat(
            points_a,
            points_b,
            cv2.FM_RANSAC,
            1.5,
            0.99
        )

        if mask_f is not None:

            fundamental_inliers = int(
                mask_f.ravel().sum()
            )

    except cv2.error:

        fundamental_inliers = 0

    # Homography
    homography_inliers = 0

    try:

        H, mask_h = cv2.findHomography(
            points_a,
            points_b,
            cv2.RANSAC,
            3.0
        )

        if mask_h is not None:

            homography_inliers = int(
                mask_h.ravel().sum()
            )

    except cv2.error:

        homography_inliers = 0

    raw = len(matches)

    return {
        "raw_matches": raw,

        "fundamental_inliers":
            fundamental_inliers,

        "fundamental_ratio":
            round(
                fundamental_inliers / raw,
                3
            ) if raw else 0,

        "homography_inliers":
            homography_inliers,

        "homography_ratio":
            round(
                homography_inliers / raw,
                3
            ) if raw else 0
    }


def main():

    print("=" * 60)
    print("3D ULPIN GEOMETRIC VERIFICATION")
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

    detector = cv2.SIFT_create(
        nfeatures=4000
    )

    results = []

    print("\nRunning RANSAC verification...\n")

    for i in range(len(images)):

        for j in range(i + 1, len(images)):

            image_a = images[i]
            image_b = images[j]

            geometry = verify_geometry(
                image_a["image"],
                image_b["image"],
                detector
            )

            result = {
                "image_a": image_a["name"],
                "image_b": image_b["name"],
                **geometry
            }

            results.append(result)

    # Rank by geometrically verified matches
    results.sort(
        key=lambda x: x["fundamental_inliers"],
        reverse=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "building01_geometric_verification.json"
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

    print("=" * 60)
    print("TOP GEOMETRICALLY VERIFIED PAIRS")
    print("=" * 60)

    for result in results[:20]:

        print(
            f"{result['image_a'][:27]:27} ↔ "
            f"{result['image_b'][:27]:27} | "
            f"raw: {result['raw_matches']:3} | "
            f"F-inliers: "
            f"{result['fundamental_inliers']:3} "
            f"({result['fundamental_ratio']:.2f}) | "
            f"H-inliers: "
            f"{result['homography_inliers']:3} "
            f"({result['homography_ratio']:.2f})"
        )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()