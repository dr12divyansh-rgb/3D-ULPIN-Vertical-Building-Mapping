from pathlib import Path
from PIL import Image, ImageDraw
import random


def create_synthetic_dataset(
    output_folder,
    number_of_images=20,
    image_size=(640, 640)
):
    """
    Generate a simple synthetic building dataset.

    Each image contains rectangular buildings.
    A corresponding binary mask is also generated.

    White in the mask = building
    Black in the mask = background
    """

    output_folder = Path(output_folder)

    images_folder = output_folder / "images"
    masks_folder = output_folder / "masks"

    images_folder.mkdir(parents=True, exist_ok=True)
    masks_folder.mkdir(parents=True, exist_ok=True)

    width, height = image_size

    for image_number in range(1, number_of_images + 1):

        # Create an aerial-like background
        image = Image.new(
            "RGB",
            image_size,
            (120, 150, 110)
        )

        image_draw = ImageDraw.Draw(image)

        # Create black mask
        mask = Image.new(
            "L",
            image_size,
            0
        )

        mask_draw = ImageDraw.Draw(mask)

        # Number of buildings
        number_of_buildings = random.randint(3, 8)

        for _ in range(number_of_buildings):

            # Random building size
            building_width = random.randint(50, 140)
            building_height = random.randint(50, 120)

            # Random position
            x1 = random.randint(
                10,
                width - building_width - 10
            )

            y1 = random.randint(
                10,
                height - building_height - 10
            )

            x2 = x1 + building_width
            y2 = y1 + building_height

            # Draw building on image
            building_color = (
                random.randint(170, 230),
                random.randint(170, 230),
                random.randint(170, 230)
            )

            image_draw.rectangle(
                [x1, y1, x2, y2],
                fill=building_color
            )

            # Draw building on mask
            mask_draw.rectangle(
                [x1, y1, x2, y2],
                fill=255
            )

        # File names
        image_name = f"building_{image_number:03d}.png"
        mask_name = f"building_{image_number:03d}_mask.png"

        # Save
        image.save(images_folder / image_name)
        mask.save(masks_folder / mask_name)

    print("=" * 50)
    print("SYNTHETIC DATASET CREATED")
    print("=" * 50)

    print(f"\nImages created: {number_of_images}")
    print(f"Image size: {width} x {height}")

    print(f"\nImages:")
    print(images_folder)

    print(f"\nMasks:")
    print(masks_folder)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    output_folder = (
        project_root
        / "data"
        / "processed"
        / "synthetic_buildings"
    )

    create_synthetic_dataset(
        output_folder=output_folder,
        number_of_images=20
    )