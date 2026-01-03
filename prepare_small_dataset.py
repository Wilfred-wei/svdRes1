#!/usr/bin/env python3
"""
Script to create a small-scale balanced dataset from CNNSpot_Split
Selects equal number of real/fake samples from each class based on percentage
"""
import os
import shutil
from pathlib import Path
import random

def setup_small_dataset(
    source_root="/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train",
    target_root="/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small",
    classes=["car", "cat", "chair", "horse"],
    percentage=0.01  # 1% of the data (default)
):
    """
    Create a small balanced dataset based on percentage

    Args:
        percentage: Fraction of data to use (e.g., 0.01 for 1%, 0.1 for 10%)
    """
    print(f"Creating small dataset at {target_root}")
    print(f"Percentage of data: {percentage * 100:.1f}%")

    # Create target directories
    for class_name in classes:
        for split in ["0_real", "1_fake"]:
            Path(target_root, class_name, split).mkdir(parents=True, exist_ok=True)

    # Copy samples
    total_copied = 0
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")

        for split in ["0_real", "1_fake"]:
            source_dir = Path(source_root) / class_name / split
            target_dir = Path(target_root) / class_name / split

            # Get all images
            images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))

            if len(images) == 0:
                print(f"  Warning: No images found in {source_dir}")
                continue

            # Calculate number of samples based on percentage
            samples_per_split = max(1, int(len(images) * percentage))

            # Ensure we don't exceed available images
            samples_per_split = min(samples_per_split, len(images))

            # Randomly select samples
            selected = random.sample(images, samples_per_split)

            # Copy files
            for img_path in selected:
                shutil.copy2(img_path, target_dir / img_path.name)
                total_copied += 1

            print(f"  {split}: Copied {len(selected)} images "
                  f"({len(selected)}/{len(images)} = {len(selected)/len(images)*100:.1f}%)")

    print(f"\n✓ Total images copied: {total_copied}")
    print(f"✓ Small dataset created at: {target_root}")
    return target_root

def setup_val_small_dataset(
    source_root="/sda/home/temp/weiwenfei/Datasets/progan_val",
    target_root="/sda/home/temp/weiwenfei/Datasets/progan_val_small",
    classes=["car", "cat", "chair", "horse"],
    percentage=0.01  # 1% of the data (default)
):
    """
    Create a small validation dataset based on percentage

    Args:
        percentage: Fraction of data to use (e.g., 0.01 for 1%, 0.1 for 10%)
    """
    print(f"\nCreating small validation dataset at {target_root}")
    print(f"Percentage of data: {percentage * 100:.1f}%")

    # Create target directories
    for class_name in classes:
        for split in ["0_real", "1_fake"]:
            Path(target_root, class_name, split).mkdir(parents=True, exist_ok=True)

    # Copy samples
    total_copied = 0
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")

        for split in ["0_real", "1_fake"]:
            source_dir = Path(source_root) / class_name / split
            target_dir = Path(target_root) / class_name / split

            if not source_dir.exists():
                print(f"  Warning: {source_dir} does not exist")
                continue

            # Get all images
            images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))

            if len(images) == 0:
                print(f"  Warning: No images found in {source_dir}")
                continue

            # Calculate number of samples based on percentage
            samples_per_split = max(1, int(len(images) * percentage))

            # Ensure we don't exceed available images
            samples_per_split = min(samples_per_split, len(images))

            # Randomly select samples
            selected = random.sample(images, samples_per_split)

            # Copy files
            for img_path in selected:
                shutil.copy2(img_path, target_dir / img_path.name)
                total_copied += 1

            print(f"  {split}: Copied {len(selected)} images "
                  f"({len(selected)}/{len(images)} = {len(selected)/len(images)*100:.1f}%)")

    print(f"\n✓ Total validation images copied: {total_copied}")
    return target_root

if __name__ == "__main__":
    random.seed(3407)

    # Setup training dataset (1% of original data)
    setup_small_dataset(
        percentage=0.01  # 1% of the data
    )

    # # Setup validation dataset (1% of original data)
    # setup_val_small_dataset(
    #     percentage=0.01  # 1% of the data
    # )
