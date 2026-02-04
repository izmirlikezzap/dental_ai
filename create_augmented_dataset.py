#!/usr/bin/env python3
"""
Apply augmentation to TRAIN SET ONLY.
Val and test sets remain original (no data leakage).

Augmentation types (5x expansion):
1. original - no changes
2. rotation +15 degrees
3. rotation -15 degrees
4. CLAHE
5. negative (invert colors)

Input: /mnt/storage_fast_8tb/datasets/jaw_dataset_rana_aug/train (pre-split)
Output: Same location (overwrite train with augmented)
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image
import albumentations as A

# Paths
DATASET_ROOT = Path("/mnt/storage_fast_8tb/datasets/jaw_dataset_rana_aug")
TRAIN_ROOT = DATASET_ROOT / "train"

TRAIN_IMAGES = TRAIN_ROOT / "images"
TRAIN_MASKS = TRAIN_ROOT / "masks"

# Temporary directory for augmented files
TEMP_AUG_ROOT = DATASET_ROOT / "train_augmented_temp"

# Classes to process (excluding healthy)
CLASSES = ['meziodens', 'supernumere']


def load_yolo_annotations(label_file: Path) -> list:
    """Load YOLO format annotations (class x_center y_center width height)"""
    if not label_file.exists():
        return []

    annotations = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                annotations.append([class_id, x_center, y_center, width, height])
    return annotations


def save_yolo_annotations(label_file: Path, annotations: list):
    """Save YOLO format annotations"""
    with open(label_file, 'w') as f:
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to pixel bbox [x_min, y_min, x_max, y_max]"""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2

    return [x_min, y_min, x_max, y_max]


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert pixel bbox to YOLO format"""
    width_px = x_max - x_min
    height_px = y_max - y_min
    x_center_px = x_min + width_px / 2
    y_center_px = y_min + height_px / 2

    x_center = x_center_px / img_width
    y_center = y_center_px / img_height
    width = width_px / img_width
    height = height_px / img_height

    return [x_center, y_center, width, height]


def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge and convert back to RGB
    lab_clahe = cv2.merge([l_clahe, a, b])
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe


def create_augmented_versions(image_path: Path, label_path: Path, output_images_dir: Path, output_labels_dir: Path):
    """
    Create 5 augmented versions of an image:
    1. original
    2. rotation +15 degrees
    3. rotation -15 degrees
    4. CLAHE
    5. negative (invert colors)
    """
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Load annotations
    annotations = load_yolo_annotations(label_path)

    # Convert YOLO to bboxes for albumentations
    bboxes = []
    class_labels = []
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        bbox = yolo_to_bbox(x_center, y_center, width, height, w, h)
        bboxes.append(bbox)
        class_labels.append(class_id)

    stem = image_path.stem

    # 1. Original (no augmentation)
    original_img_path = output_images_dir / f"{stem}_original.jpg"
    original_lbl_path = output_labels_dir / f"{stem}_original.txt"

    cv2.imwrite(str(original_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    save_yolo_annotations(original_lbl_path, annotations)

    # 2. Rotation +15 degrees
    transform_rot_p15 = A.Compose([
        A.Rotate(limit=(15, 15), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    if bboxes:
        transformed = transform_rot_p15(image=image, bboxes=bboxes, class_labels=class_labels)
        rot_p15_img = transformed['image']
        rot_p15_bboxes = transformed['bboxes']
        rot_p15_labels = transformed['class_labels']

        # Convert back to YOLO format
        rot_p15_annotations = []
        for bbox, class_id in zip(rot_p15_bboxes, rot_p15_labels):
            x_min, y_min, x_max, y_max = bbox
            yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, w, h)
            rot_p15_annotations.append([class_id] + yolo_bbox)
    else:
        # No bboxes - just transform image
        rot_p15_img = transform_rot_p15(image=image)['image']
        rot_p15_annotations = []

    rot_p15_img_path = output_images_dir / f"{stem}_rot_p15.jpg"
    rot_p15_lbl_path = output_labels_dir / f"{stem}_rot_p15.txt"
    cv2.imwrite(str(rot_p15_img_path), cv2.cvtColor(rot_p15_img, cv2.COLOR_RGB2BGR))
    save_yolo_annotations(rot_p15_lbl_path, rot_p15_annotations)

    # 3. Rotation -15 degrees
    transform_rot_n15 = A.Compose([
        A.Rotate(limit=(-15, -15), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    if bboxes:
        transformed = transform_rot_n15(image=image, bboxes=bboxes, class_labels=class_labels)
        rot_n15_img = transformed['image']
        rot_n15_bboxes = transformed['bboxes']
        rot_n15_labels = transformed['class_labels']

        rot_n15_annotations = []
        for bbox, class_id in zip(rot_n15_bboxes, rot_n15_labels):
            x_min, y_min, x_max, y_max = bbox
            yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, w, h)
            rot_n15_annotations.append([class_id] + yolo_bbox)
    else:
        rot_n15_img = transform_rot_n15(image=image)['image']
        rot_n15_annotations = []

    rot_n15_img_path = output_images_dir / f"{stem}_rot_n15.jpg"
    rot_n15_lbl_path = output_labels_dir / f"{stem}_rot_n15.txt"
    cv2.imwrite(str(rot_n15_img_path), cv2.cvtColor(rot_n15_img, cv2.COLOR_RGB2BGR))
    save_yolo_annotations(rot_n15_lbl_path, rot_n15_annotations)

    # 4. CLAHE (no bbox transformation needed)
    clahe_img = apply_clahe(image)
    clahe_img_path = output_images_dir / f"{stem}_clahe.jpg"
    clahe_lbl_path = output_labels_dir / f"{stem}_clahe.txt"
    cv2.imwrite(str(clahe_img_path), cv2.cvtColor(clahe_img, cv2.COLOR_RGB2BGR))
    save_yolo_annotations(clahe_lbl_path, annotations)

    # 5. Negative (invert colors - no bbox transformation needed)
    negative_img = 255 - image
    negative_img_path = output_images_dir / f"{stem}_negative.jpg"
    negative_lbl_path = output_labels_dir / f"{stem}_negative.txt"
    cv2.imwrite(str(negative_img_path), cv2.cvtColor(negative_img, cv2.COLOR_RGB2BGR))
    save_yolo_annotations(negative_lbl_path, annotations)

    return 5  # Number of augmented versions created


def main():
    print("="*80)
    print("TRAIN SET AUGMENTATION (5x expansion)")
    print("="*80)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Augmenting: train set only")
    print(f"Val/Test: remain original")
    print(f"Classes: {CLASSES}")
    print(f"Augmentation: 5x (original, rot±15°, CLAHE, negative)")
    print("="*80)

    # Check if train directory exists
    if not TRAIN_ROOT.exists():
        print(f"ERROR: Train directory not found: {TRAIN_ROOT}")
        print("Please run prepare_split_dataset.py first!")
        return

    # Create temporary augmented directory
    TEMP_AUG_ROOT.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        print(f"\nProcessing class: {class_name}")

        # Create temporary augmented directories
        temp_images_dir = TEMP_AUG_ROOT / "images" / class_name
        temp_masks_dir = TEMP_AUG_ROOT / "masks" / class_name

        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_masks_dir.mkdir(parents=True, exist_ok=True)

        # Get all images from original train set
        source_images_dir = TRAIN_IMAGES / class_name
        source_masks_dir = TRAIN_MASKS / class_name

        if not source_images_dir.exists():
            print(f"  WARNING: {source_images_dir} not found, skipping")
            continue

        image_files = list(source_images_dir.glob("*.jpg")) + \
                      list(source_images_dir.glob("*.png")) + \
                      list(source_images_dir.glob("*.jpeg"))

        print(f"  Found {len(image_files)} original images")
        print(f"  Creating 5x augmented versions ({len(image_files) * 5} total images)")

        total_augmented = 0
        for image_path in tqdm(image_files, desc=f"  Augmenting {class_name}"):
            # Find corresponding label file
            label_path = source_masks_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"    WARNING: Label not found for {image_path.name}, skipping")
                continue

            # Create 5 augmented versions
            num_created = create_augmented_versions(
                image_path, label_path,
                temp_images_dir, temp_masks_dir
            )
            total_augmented += num_created

        print(f"  Created {total_augmented} augmented images")

    print("\n" + "="*80)
    print("REPLACING TRAIN SET WITH AUGMENTED VERSION")
    print("="*80)

    # Backup original train set
    backup_dir = DATASET_ROOT / "train_original_backup"
    if backup_dir.exists():
        print("Removing old backup...")
        shutil.rmtree(backup_dir)

    print(f"Backing up original train to: {backup_dir}")
    shutil.copytree(TRAIN_ROOT, backup_dir)

    # Remove original train
    print(f"Removing original train...")
    shutil.rmtree(TRAIN_ROOT)

    # Move augmented to train
    print(f"Moving augmented to train...")
    shutil.move(TEMP_AUG_ROOT, TRAIN_ROOT)

    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)

    # Print summary
    print("\nFinal dataset structure:")
    for split in ['train', 'val', 'test']:
        split_dir = DATASET_ROOT / split / "images"
        if split_dir.exists():
            total_count = 0
            print(f"\n{split.upper()}:")
            for class_name in CLASSES:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    n_images = len(list(class_dir.glob("*")))
                    total_count += n_images
                    print(f"  {class_name}: {n_images} images")
            print(f"  Total: {total_count} images")
            if split == 'train':
                print(f"  (5x augmented from {total_count // 5} originals)")

    print(f"\nDataset ready at: {DATASET_ROOT}")
    print(f"Original train backed up at: {backup_dir}")


if __name__ == "__main__":
    main()
