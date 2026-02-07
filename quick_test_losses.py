#!/usr/bin/env python3
"""
Quick Test: Compare Default vs Focal Loss
Train yolov5n for 20 epochs with both losses to verify focal loss is working
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# Set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from yolo_detection_train_single_gpu import Trainer, get_all_images
from experiment_config import ExperimentManager, create_experiment_config
from sklearn.model_selection import KFold
import numpy as np

# Quick test configuration
QUICK_TEST_DIR = Path("/home/eva/PycharmProjects/dentalAI/yolo_detection_results_quick_test")
MODEL = "yolov5n"
MODEL_SIZE = "n"
EPOCHS = 20  # Quick test
BATCH_SIZE = 32
IMAGE_SIZE = 1024
FOLD = 0  # Only test fold 0

def quick_test_single_loss(loss_method: str, gpu_id: int = 0):
    """Run quick test for a single loss method"""
    print("\n" + "=" * 80)
    print(f"QUICK TEST: {MODEL} with {loss_method.upper()} loss ({EPOCHS} epochs)")
    print("=" * 80)

    # Create experiment config
    config = create_experiment_config(
        model_name=MODEL,
        model_size=MODEL_SIZE,
        loss_method=loss_method,
        is_augmented=False,  # No augmentation for quick test
        n_folds=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        patience=20,  # No early stopping for quick test
        device=str(gpu_id)
    )

    # Create experiment manager
    manager = ExperimentManager(QUICK_TEST_DIR)

    # Get images and split
    all_images = get_all_images()
    print(f"[INFO] Total images: {len(all_images)}")

    # Use KFold to get train/val split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(all_images))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if fold_idx != FOLD:
            continue

        train_images = [all_images[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]

        print(f"[INFO] Fold {FOLD}: {len(train_images)} train, {len(val_images)} val")

        # Create trainer
        trainer = Trainer(config, manager, device=str(gpu_id))

        # Train
        fold_results = trainer.train_fold(FOLD, train_images, val_images)

        # Update results
        manager.update_results(
            experiment_id=config.experiment_id,
            fold=FOLD,
            results=fold_results
        )

        print(f"\n[RESULTS] {loss_method.upper()}")
        print(f"  mAP50:     {fold_results['metrics']['mAP50']:.4f}")
        print(f"  mAP50-95:  {fold_results['metrics']['mAP50-95']:.4f}")
        print(f"  Precision: {fold_results['metrics']['precision']:.4f}")
        print(f"  Recall:    {fold_results['metrics']['recall']:.4f}")
        print(f"  Time:      {fold_results['training_time_seconds']/60:.1f} min")

        return fold_results


def compare_results():
    """Compare results from both tests"""
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)

    master_file = QUICK_TEST_DIR / "master_results.json"
    if not master_file.exists():
        print("[ERROR] No results found!")
        return

    with open(master_file, 'r') as f:
        results = json.load(f)

    # Find default and focal results
    default_res = None
    focal_res = None

    for exp_id, exp_data in results.items():
        if 'default' in exp_id and 'normal' in exp_id:
            default_res = exp_data['folds']['fold_0']['metrics']
        elif 'focal' in exp_id and 'normal' in exp_id:
            focal_res = exp_data['folds']['fold_0']['metrics']

    if not default_res or not focal_res:
        print("[ERROR] Could not find both results!")
        print(f"Default: {default_res is not None}")
        print(f"Focal: {focal_res is not None}")
        return

    print(f"\n{'Metric':<15} {'Default':<10} {'Focal':<10} {'Diff':<10}")
    print("-" * 50)

    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    for metric in metrics:
        default_val = default_res.get(metric, 0)
        focal_val = focal_res.get(metric, 0)
        diff = focal_val - default_val
        sign = "+" if diff > 0 else ""

        print(f"{metric:<15} {default_val:<10.4f} {focal_val:<10.4f} {sign}{diff:<10.4f}")

    # Check if focal loss is actually different
    if abs(default_res['mAP50'] - focal_res['mAP50']) < 0.0001:
        print("\n⚠️  WARNING: Results are identical! Focal loss may not be working.")
    else:
        print("\n✅ Results differ! Focal loss is active.")


def main():
    print("=" * 80)
    print("QUICK LOSS COMPARISON TEST")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Fold: {FOLD}")
    print(f"Results: {QUICK_TEST_DIR}")
    print("=" * 80)

    # Test 1: Default loss
    print("\n[TEST 1/2] Training with DEFAULT loss...")
    default_results = quick_test_single_loss('default', gpu_id=0)

    # Test 2: Focal loss
    print("\n[TEST 2/2] Training with FOCAL loss...")
    focal_results = quick_test_single_loss('focal', gpu_id=0)

    # Compare
    compare_results()

    print("\n" + "=" * 80)
    print("QUICK TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: {QUICK_TEST_DIR}")
    print("\nTo view detailed training curves:")
    print(f"  Default: {QUICK_TEST_DIR}/normal/default_loss/runs/")
    print(f"  Focal:   {QUICK_TEST_DIR}/normal/focal_loss/runs/")


if __name__ == "__main__":
    main()
