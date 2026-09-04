from pathlib import Path
import json


def pixel_to_geographic(
    pixel_x,
    pixel_y,
    origin_x,
    origin_y,
    pixel_width,
    pixel_height
):
    """
    Convert pixel coordinates into geographic coordinates.

    This is a simple affine transformation.

    Parameters:
        pixel_x       : X coordinate of pixel
        pixel_y       : Y coordinate of pixel
        origin_x      : geographic X coordinate of image origin
        origin_y      : geographic Y coordinate of image origin
        pixel_width   : geographic width represented by one pixel
        pixel_height  : geographic height represented by one pixel

    Returns:
        geographic_x, geographic_y
    """

    geographic_x = (
        origin_x
        + pixel_x * pixel_width
    )

    geographic_y = (
        origin_y
        + pixel_y * pixel_height
    )

    return geographic_x, geographic_y


def convert_polygon(
    polygon,
    origin_x,
    origin_y,
    pixel_width,
    pixel_height
):
    """
    Convert an entire polygon from
    pixel coordinates to geographic coordinates.
    """

    geographic_polygon = []

    for point in polygon:

        pixel_x = point[0]
        pixel_y = point[1]

        geographic_x, geographic_y = (
            pixel_to_geographic(
                pixel_x,
                pixel_y,
                origin_x,
                origin_y,
                pixel_width,
                pixel_height
            )
        )

        geographic_polygon.append([
            geographic_x,
            geographic_y
        ])

    return geographic_polygon


if __name__ == "__main__":

    print("=" * 60)
    print("PIXEL TO GEOGRAPHIC COORDINATE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # TEST VALUES
    # --------------------------------------------------------
    #
    # These are ONLY example values.
    # They are NOT from your actual dataset.
    #

    origin_x = 79.1300
    origin_y = 12.9200

    pixel_width = 0.00001
    pixel_height = -0.00001

    # Example polygon
    polygon = [
        [81, 59],
        [80, 75],
        [90, 82],
        [91, 103],
        [100, 113],
        [114, 113],
        [116, 86],
        [100, 80],
        [99, 59]
    ]

    geographic_polygon = convert_polygon(
        polygon,
        origin_x,
        origin_y,
        pixel_width,
        pixel_height
    )

    print("\nPixel polygon:")

    print(polygon)

    print("\nGeographic polygon:")

    for point in geographic_polygon:

        print(
            f"Longitude: {point[0]:.7f}, "
            f"Latitude: {point[1]:.7f}"
        )

    print("\nCoordinate conversion test successful!")