from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class BuildingDataset(Dataset):

    def __init__(self, image_files, mask_files):

        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load mask
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert to NumPy
        image = np.array(image)
        mask = np.array(mask)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Convert image:
        # H x W x C
        # into:
        # C x H x W
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Convert mask to 0 or 1
        mask = (mask > 127).astype(np.float32)

        # Add channel dimension
        mask = np.expand_dims(
            mask,
            axis=0
        )

        return (
            torch.tensor(image),
            torch.tensor(mask)
        )


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, input_channels, output_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(3, 16)
        self.encoder2 = DoubleConv(16, 32)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            64,
            32
        )

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            32,
            16
        )

        # Output
        self.output = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool(e1)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e2)
        )

        # Decoder
        d2 = self.up2(b)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.decoder2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.decoder1(d1)

        # Output
        output = self.output(d1)

        return output


# ============================================================
# DATA PREPARATION
# ============================================================

def load_dataset():

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    image_folder = (
        project_root
        / "data"
        / "processed"
        / "synthetic_buildings"
        / "images"
    )

    mask_folder = (
        project_root
        / "data"
        / "processed"
        / "synthetic_buildings"
        / "masks"
    )

    image_files = sorted(
        image_folder.glob("*.png")
    )

    mask_files = []

    valid_images = []

    for image_file in image_files:

        mask_name = (
            image_file.stem
            + "_mask.png"
        )

        mask_file = (
            mask_folder
            / mask_name
        )

        if mask_file.exists():

            valid_images.append(
                image_file
            )

            mask_files.append(
                mask_file
            )

    return valid_images, mask_files


# ============================================================
# TRAINING
# ============================================================

def train_model():

    #Find the project root
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    print("=" * 60)
    print("BUILDING SEGMENTATION TRAINING")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # Load dataset
    images, masks = load_dataset()

    print("\nTotal image-mask pairs:", len(images))

    if len(images) == 0:

        print("\nERROR: No training data found.")

        return

    # Combine and shuffle
    combined = list(
        zip(images, masks)
    )

    random.seed(42)

    random.shuffle(combined)

    images, masks = zip(*combined)

    images = list(images)
    masks = list(masks)

    # Train/validation split
    split_index = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:split_index]
    train_masks = masks[:split_index]

    val_images = images[split_index:]
    val_masks = masks[split_index:]

    print("Training samples:", len(train_images))
    print("Validation samples:", len(val_images))

    # Dataset objects
    train_dataset = BuildingDataset(
        train_images,
        train_masks
    )

    val_dataset = BuildingDataset(
        val_images,
        val_masks
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Create model
    model = UNet().to(DEVICE)

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting training...\n")

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        total_training_loss = 0

        for images_batch, masks_batch in train_loader:

            images_batch = images_batch.to(DEVICE)
            masks_batch = masks_batch.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Prediction
            predictions = model(
                images_batch
            )

            # Calculate loss
            loss = loss_function(
                predictions,
                masks_batch
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_training_loss += loss.item()

        average_training_loss = (
            total_training_loss
            / len(train_loader)
        )

        # Validation
        model.eval()

        total_validation_loss = 0

        with torch.no_grad():

            for images_batch, masks_batch in val_loader:

                images_batch = images_batch.to(DEVICE)
                masks_batch = masks_batch.to(DEVICE)

                predictions = model(
                    images_batch
                )

                loss = loss_function(
                    predictions,
                    masks_batch
                )

                total_validation_loss += loss.item()

        average_validation_loss = (
            total_validation_loss
            / len(val_loader)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Training Loss: {average_training_loss:.4f} "
            f"| Validation Loss: {average_validation_loss:.4f}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_folder = (
        project_root
        / "models"
        / "building_detection"
    )

    model_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_folder
        / "building_model.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nModel saved to:")

    print(model_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()