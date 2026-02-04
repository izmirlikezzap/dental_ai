#!/usr/bin/env python3
"""
YOLO Detection Training for Dental X-ray Analysis
==================================================

Trains all YOLO models on jaw dataset for meziodens and supernumere detection.
Compares model performance and saves results.

Dataset:
- healthy: 140 images (background, no annotations)
- meziodens: 199 images (class 0)
- supernumere: 78 images (class 1)

Usage:
    python classification_train.py

    # With custom tag
    RUN_TAG=exp2 python classification_train.py

    # Specify GPU
    CUDA_VISIBLE_DEVICES=0,1 python classification_train.py
"""

import os
import sys
import json
import shutil
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ultralytics YOLO
from ultralytics import YOLO

from utility.classification_models import (
    get_yolo_detection_models,
    NUM_CLASSES,
    CLASS_NAMES
)
from utility.detection_metrics import (
    calculate_metrics_from_confusion_matrix,
    print_metrics_report
)


# ==================== CONFIGURATION ====================

RUN_TAG = os.environ.get("RUN_TAG", "yolo_detection_v3")

# Dataset paths
DATASET_ROOT = Path("/mnt/storage_fast_8tb/datasets/jaw_dataset_rana")
IMAGES_ROOT = DATASET_ROOT / "images"
MASKS_ROOT = DATASET_ROOT / "masks"

# Output paths
OUTPUT_ROOT = Path(f"/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_{RUN_TAG}")
RESULTS_DIR = Path(f"{RUN_TAG}_results")

# Training configuration - Optimized for better recall and class balance
TRAIN_CONFIG = {
    'epochs': 150,
    'patience': 30,
    'batch': 16,
    'imgsz': 640,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # Loss weights - Increased cls for better classification
    'box': 7.5,
    'cls': 1.5,        # Increased from 0.5 - more focus on classification
    'dfl': 1.5,

    # NO Augmentation
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'degrees': 0.0,
    'translate': 0.0,
    'scale': 0.0,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.0,
    'mosaic': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
}


# ==================== DATASET PREPARATION ====================

def prepare_yolo_dataset(
    images_root: Path,
    masks_root: Path,
    output_root: Path,
    val_split: float = 0.2,
    seed: int = 42
) -> Path:
    """
    Prepare dataset in YOLO format with train/val split.

    Args:
        images_root: Path to images directory (with class subdirs)
        masks_root: Path to masks directory (with class subdirs)
        output_root: Path to output YOLO formatted dataset
        val_split: Validation split ratio
        seed: Random seed for reproducibility

    Returns:
        Path to created data.yaml file
    """
    print(f"\n{'='*60}")
    print("PREPARING YOLO DATASET")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)

    # Create output directories
    train_images = output_root / "images" / "train"
    val_images = output_root / "images" / "val"
    train_labels = output_root / "labels" / "train"
    val_labels = output_root / "labels" / "val"

    for dir_path in [train_images, val_images, train_labels, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Collect all samples - exclude healthy (empty annotations cause high FP)
    all_samples = []  # (image_path, label_path, class_name)

    for class_name in ['meziodens', 'supernumere']:  # Healthy excluded
        class_images = images_root / class_name
        class_masks = masks_root / class_name

        if not class_images.exists():
            print(f"WARNING: {class_images} does not exist")
            continue

        for img_file in class_images.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = class_masks / f"{img_file.stem}.txt"
                if label_file.exists():
                    all_samples.append((img_file, label_file, class_name))

    print(f"Total samples found: {len(all_samples)} (healthy excluded)")

    # Split by class to maintain distribution
    train_samples = []
    val_samples = []

    for class_name in ['meziodens', 'supernumere']:  # Healthy excluded
        class_samples = [s for s in all_samples if s[2] == class_name]

        if len(class_samples) == 0:
            continue

        # Stratified split
        n_val = max(1, int(len(class_samples) * val_split))
        random.shuffle(class_samples)

        val_samples.extend(class_samples[:n_val])
        train_samples.extend(class_samples[n_val:])

        print(f"  {class_name}: {len(class_samples)} total -> {len(class_samples) - n_val} train, {n_val} val")

    # Copy files to output directories
    print("\nCopying files...")

    def copy_samples(samples: List[Tuple], img_dir: Path, lbl_dir: Path, desc: str):
        for img_path, lbl_path, class_name in tqdm(samples, desc=desc):
            # Copy image
            dst_img = img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy label
            dst_lbl = lbl_dir / lbl_path.name
            if not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)

    copy_samples(train_samples, train_images, train_labels, "Train")
    copy_samples(val_samples, val_images, val_labels, "Val")

    # Create data.yaml
    data_yaml = output_root / "data.yaml"
    data_config = {
        'path': str(output_root),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'meziodens', 1: 'supernumere'},
        'nc': 2
    }

    with open(data_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"\nDataset prepared at: {output_root}")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  data.yaml: {data_yaml}")

    return data_yaml


# ==================== RESULTS READING ====================

def read_training_results(csv_path: Path) -> Tuple[Dict, int, int]:
    """
    Read training results from CSV file.

    Args:
        csv_path: Path to results.csv

    Returns:
        Tuple of (metrics_dict, best_epoch, total_epochs)
    """
    metrics = {
        'mAP50': 0.0,
        'mAP50-95': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }
    best_epoch = -1
    epochs_trained = 0

    if not csv_path.exists():
        return metrics, best_epoch, epochs_trained

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if len(df) == 0:
            return metrics, best_epoch, epochs_trained

        epochs_trained = len(df)

        # Find best epoch by mAP50-95
        map_col = [c for c in df.columns if 'mAP50-95' in c]
        if map_col:
            best_idx = df[map_col[0]].idxmax()
            best_row = df.iloc[best_idx]
            best_epoch = int(best_row['epoch']) if 'epoch' in df.columns else best_idx + 1

            # Extract metrics from best epoch
            for col in df.columns:
                col_stripped = col.strip()
                if 'mAP50(B)' in col_stripped and '95' not in col_stripped:
                    metrics['mAP50'] = float(best_row[col])
                elif 'mAP50-95(B)' in col_stripped:
                    metrics['mAP50-95'] = float(best_row[col])
                elif 'precision(B)' in col_stripped:
                    metrics['precision'] = float(best_row[col])
                elif 'recall(B)' in col_stripped:
                    metrics['recall'] = float(best_row[col])

    except Exception as e:
        print(f"  Warning: Could not read CSV results: {e}")

    return metrics, best_epoch, epochs_trained


# ==================== DETAILED METRICS ====================

def calculate_detailed_metrics(
    model_path: str,
    data_yaml: Path,
    device: str,
    class_names: Dict[int, str]
) -> Dict:
    """
    Calculate detailed metrics by running validation on trained model.

    Args:
        model_path: Path to best.pt weights
        data_yaml: Path to data.yaml
        device: CUDA device
        class_names: Dict mapping class_id to name

    Returns:
        Dictionary with detailed metrics including IoU, Dice, Sensitivity, Specificity
    """
    try:
        # Load trained model
        model = YOLO(model_path)

        # Run validation - plots=True required for confusion matrix population
        val_results = model.val(
            data=str(data_yaml),
            device=device,
            verbose=False,
            plots=True
        )

        # Initialize metrics dict
        detailed_metrics = {
            'per_class': {},
            'overall': {}
        }

        # Get confusion matrix from validation
        if hasattr(val_results, 'confusion_matrix'):
            cm = val_results.confusion_matrix.matrix

            # Calculate metrics from confusion matrix
            num_classes = len(class_names)

            total_tp = 0
            total_fp = 0
            total_fn = 0
            all_ious = []
            all_f1s = []
            all_precisions = []
            all_recalls = []
            all_sensitivities = []
            all_specificities = []

            for class_id in range(num_classes):
                class_name = class_names.get(class_id, str(class_id))

                # TP: diagonal element
                tp = cm[class_id, class_id] if class_id < cm.shape[0] else 0

                # FP: sum of column minus TP
                fp = cm[:, class_id].sum() - tp if class_id < cm.shape[1] else 0

                # FN: sum of row minus TP (including background predictions)
                fn = cm[class_id, :].sum() - tp if class_id < cm.shape[0] else 0

                # TN: other classes correct predictions
                tn = cm.trace() - tp

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                sensitivity = recall
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_sensitivities.append(sensitivity)
                all_specificities.append(specificity)
                all_f1s.append(f1)
                all_ious.append(iou)

                detailed_metrics['per_class'][class_name] = {
                    'TP': int(tp),
                    'FP': int(fp),
                    'FN': int(fn),
                    'TN': int(tn),
                    'Precision': float(precision),
                    'Recall': float(recall),
                    'Sensitivity': float(sensitivity),
                    'Specificity': float(specificity),
                    'F1_Score': float(f1),
                    'Dice': float(f1),
                    'IoU': float(iou),
                    'Support': int(tp + fn)
                }

            # Overall metrics
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

            detailed_metrics['overall'] = {
                'Total_TP': int(total_tp),
                'Total_FP': int(total_fp),
                'Total_FN': int(total_fn),
                'Macro_Precision': float(np.mean(all_precisions)),
                'Macro_Recall': float(np.mean(all_recalls)),
                'Macro_F1_Score': float(np.mean(all_f1s)),
                'Macro_Dice': float(np.mean(all_f1s)),
                'Macro_IoU': float(np.mean(all_ious)),
                'Macro_Sensitivity': float(np.mean(all_sensitivities)),
                'Macro_Specificity': float(np.mean(all_specificities)),
                'Micro_Precision': float(micro_precision),
                'Micro_Recall': float(micro_recall),
                'Micro_F1_Score': float(micro_f1),
                'Mean_IoU': float(np.mean(all_ious)),
            }

        # Add per-class AP if available
        if hasattr(val_results.box, 'ap50'):
            for i, ap in enumerate(val_results.box.ap50):
                class_name = class_names.get(i, str(i))
                if class_name in detailed_metrics['per_class']:
                    detailed_metrics['per_class'][class_name]['AP50'] = float(ap)

        if hasattr(val_results.box, 'ap'):
            for i, ap in enumerate(val_results.box.ap):
                class_name = class_names.get(i, str(i))
                if class_name in detailed_metrics['per_class']:
                    detailed_metrics['per_class'][class_name]['AP50_95'] = float(ap)

        return detailed_metrics

    except Exception as e:
        print(f"  Warning: Could not calculate detailed metrics: {e}")
        return {}


# ==================== TRAINING ====================

def train_single_model(
    model_name: str,
    model_config: Dict,
    data_yaml: Path,
    results_dir: Path,
    device: str,
    train_config: Dict,
    master_data: Dict,
    master_file: Path
) -> Optional[Dict]:
    """
    Train a single YOLO model.

    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        data_yaml: Path to data.yaml
        results_dir: Directory to save results
        device: CUDA device(s) to use
        train_config: Training hyperparameters
        master_data: Master results dictionary to update
        master_file: Path to master results JSON file

    Returns:
        Training results dictionary or None if failed
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    print(f"  Model: {model_config['description']}")
    print(f"  Parameters: {model_config['params']}")
    print(f"  Device: {device}")

    # Check if already trained
    if model_name in master_data.get('models', {}) and \
       master_data['models'][model_name].get('status') == 'completed':
        print(f"  Skipping - already trained")
        return master_data['models'][model_name].get('results')

    # Initialize model entry
    if model_name not in master_data['models']:
        master_data['models'][model_name] = {
            'config': model_config,
            'status': 'starting',
            'started_at': datetime.now().isoformat()
        }

    try:
        # Load model
        model = YOLO(model_config['model_path'])

        # Project directory for this model
        project_dir = results_dir / "runs"

        # Update status
        master_data['models'][model_name]['status'] = 'training'

        # Train
        results = model.train(
            data=str(data_yaml),
            epochs=train_config['epochs'],
            patience=train_config['patience'],
            batch=train_config['batch'],
            imgsz=train_config['imgsz'],
            device=device,
            project=str(project_dir),
            name=model_name,
            exist_ok=True,
            pretrained=True,
            optimizer=train_config['optimizer'],
            lr0=train_config['lr0'],
            lrf=train_config['lrf'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay'],
            warmup_epochs=train_config['warmup_epochs'],
            warmup_momentum=train_config['warmup_momentum'],
            warmup_bias_lr=train_config['warmup_bias_lr'],
            box=train_config['box'],
            cls=train_config['cls'],
            dfl=train_config['dfl'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            degrees=train_config['degrees'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            shear=train_config['shear'],
            perspective=train_config['perspective'],
            flipud=train_config['flipud'],
            fliplr=train_config['fliplr'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
            copy_paste=train_config['copy_paste'],
            save=True,
            save_period=-1,
            plots=True,
            verbose=True,
            workers=2,
        )

        # Read metrics from CSV (most reliable source)
        csv_path = project_dir / model_name / "results.csv"
        metrics, best_epoch, epochs_trained = read_training_results(csv_path)

        # Calculate detailed metrics from validation
        detailed_metrics = calculate_detailed_metrics(
            model_path=str(project_dir / model_name / "weights" / "best.pt"),
            data_yaml=data_yaml,
            device=device,
            class_names=CLASS_NAMES
        )

        # Merge metrics
        metrics.update(detailed_metrics)

        print(f"\n  {'='*50}")
        print(f"  {model_name} DETAILED RESULTS")
        print(f"  {'='*50}")
        print(f"  Best Epoch: {best_epoch}/{epochs_trained}")
        print(f"\n  --- Standard Metrics ---")
        print(f"  mAP50: {metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        if 'per_class' in metrics:
            print(f"\n  --- Per-Class Metrics ---")
            print(f"  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1/Dice':>8} {'IoU':>8} {'Sens':>8} {'Spec':>8}")
            print(f"  {'-'*68}")
            for cls_name, cls_metrics in metrics['per_class'].items():
                print(f"  {cls_name:<12} "
                      f"{cls_metrics.get('Precision', 0):>8.4f} "
                      f"{cls_metrics.get('Recall', 0):>8.4f} "
                      f"{cls_metrics.get('F1_Score', 0):>8.4f} "
                      f"{cls_metrics.get('IoU', 0):>8.4f} "
                      f"{cls_metrics.get('Sensitivity', 0):>8.4f} "
                      f"{cls_metrics.get('Specificity', 0):>8.4f}")

        if 'overall' in metrics:
            print(f"\n  --- Overall Metrics (Macro) ---")
            overall = metrics['overall']
            print(f"  Macro Precision:  {overall.get('Macro_Precision', 0):.4f}")
            print(f"  Macro Recall:     {overall.get('Macro_Recall', 0):.4f}")
            print(f"  Macro F1/Dice:    {overall.get('Macro_F1_Score', 0):.4f}")
            print(f"  Macro IoU (mIoU): {overall.get('Macro_IoU', 0):.4f}")
            print(f"  Macro Sensitivity:{overall.get('Macro_Sensitivity', 0):.4f}")
            print(f"  Macro Specificity:{overall.get('Macro_Specificity', 0):.4f}")

        print(f"  {'='*50}")

        # Get best metrics from results
        result_data = {
            'model_name': model_name,
            'model_family': model_config['family'],
            'params': model_config['params'],
            'epochs_trained': epochs_trained,
            'best_epoch': best_epoch,
            'metrics': metrics,
            'weights_path': str(project_dir / model_name / "weights" / "best.pt"),
            'completed_at': datetime.now().isoformat()
        }

        # Update master data
        master_data['models'][model_name]['status'] = 'completed'
        master_data['models'][model_name]['results'] = result_data
        master_data['models'][model_name]['completed_at'] = datetime.now().isoformat()

        # Save immediately after each model completes
        save_master_results(master_file, master_data)
        print(f"\n  Results saved to {master_file}")

        return result_data

    except Exception as e:
        print(f"\n  ERROR: {e}")
        master_data['models'][model_name]['status'] = 'failed'
        master_data['models'][model_name]['error'] = str(e)
        master_data['models'][model_name]['failed_at'] = datetime.now().isoformat()

        # Save error state immediately
        save_master_results(master_file, master_data)
        print(f"  Error state saved to {master_file}")

        import traceback
        traceback.print_exc()

        return None


def save_master_results(master_file: Path, data: Dict):
    """Save master results to JSON file."""
    with open(master_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_master_results(master_file: Path) -> Dict:
    """Load master results from JSON file or create new."""
    if master_file.exists():
        with open(master_file, 'r') as f:
            return json.load(f)
    return {
        'experiment_info': {
            'name': 'YOLO Detection Training',
            'description': 'Dental X-ray meziodens/supernumere detection',
            'run_tag': RUN_TAG,
            'created_at': datetime.now().isoformat(),
            'dataset_root': str(DATASET_ROOT),
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
        },
        'training_config': TRAIN_CONFIG,
        'models': {}
    }


def create_comparison_report(master_data: Dict, results_dir: Path):
    """Create comparison report of all trained models with detailed metrics."""
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE COMPARISON REPORT")
    print(f"{'='*80}")

    # Collect results
    rows = []
    for model_name, model_data in master_data['models'].items():
        if model_data.get('status') != 'completed':
            continue

        results = model_data.get('results', {})
        metrics = results.get('metrics', {})
        overall = metrics.get('overall', {})

        row = {
            'Model': model_name,
            'Family': results.get('model_family', ''),
            'Params': results.get('params', ''),
            'mAP50': metrics.get('mAP50', 0),
            'mAP50-95': metrics.get('mAP50-95', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'Macro_F1_Dice': overall.get('Macro_F1_Score', 0),
            'Macro_IoU': overall.get('Macro_IoU', 0),
            'Macro_Sensitivity': overall.get('Macro_Sensitivity', 0),
            'Macro_Specificity': overall.get('Macro_Specificity', 0),
            'Epochs': results.get('epochs_trained', 0),
            'Best_Epoch': results.get('best_epoch', -1),
        }

        # Add per-class metrics
        per_class = metrics.get('per_class', {})
        for cls_name, cls_metrics in per_class.items():
            row[f'{cls_name}_F1'] = cls_metrics.get('F1_Score', 0)
            row[f'{cls_name}_IoU'] = cls_metrics.get('IoU', 0)
            row[f'{cls_name}_Precision'] = cls_metrics.get('Precision', 0)
            row[f'{cls_name}_Recall'] = cls_metrics.get('Recall', 0)

        rows.append(row)

    if not rows:
        print("No completed models to compare")
        return

    # Create DataFrame and sort by mAP50-95
    df = pd.DataFrame(rows)
    df = df.sort_values('mAP50-95', ascending=False)

    # Save to CSV
    csv_path = results_dir / f"{RUN_TAG}_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Comparison saved to: {csv_path}")

    # Print top models with all metrics
    print("\n" + "=" * 120)
    print("TOP 10 MODELS BY mAP50-95 (with all metrics)")
    print("=" * 120)
    print(f"{'Model':<12} {'mAP50':>8} {'mAP50-95':>10} {'F1/Dice':>10} {'mIoU':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 120)
    for _, row in df.head(10).iterrows():
        print(f"{row['Model']:<12} "
              f"{row['mAP50']:>8.4f} "
              f"{row['mAP50-95']:>10.4f} "
              f"{row.get('Macro_F1_Dice', 0):>10.4f} "
              f"{row.get('Macro_IoU', 0):>8.4f} "
              f"{row.get('Macro_Sensitivity', 0):>8.4f} "
              f"{row.get('Macro_Specificity', 0):>8.4f} "
              f"{row['Precision']:>8.4f} "
              f"{row['Recall']:>8.4f}")

    # Best model per family
    print("\n" + "=" * 120)
    print("BEST MODEL PER FAMILY")
    print("=" * 120)
    for family in df['Family'].unique():
        family_df = df[df['Family'] == family]
        if len(family_df) > 0:
            best = family_df.iloc[0]
            print(f"{family:<10} -> {best['Model']:<12} | mAP50-95: {best['mAP50-95']:.4f} | "
                  f"F1/Dice: {best.get('Macro_F1_Dice', 0):.4f} | mIoU: {best.get('Macro_IoU', 0):.4f}")

    # Per-class best models
    print("\n" + "=" * 120)
    print("BEST MODELS PER CLASS")
    print("=" * 120)
    for cls_name in ['meziodens', 'supernumere']:
        f1_col = f'{cls_name}_F1'
        if f1_col in df.columns:
            df_sorted = df.sort_values(f1_col, ascending=False)
            best = df_sorted.iloc[0]
            print(f"{cls_name:<12} -> {best['Model']:<12} | "
                  f"F1: {best.get(f1_col, 0):.4f} | "
                  f"IoU: {best.get(f'{cls_name}_IoU', 0):.4f} | "
                  f"Precision: {best.get(f'{cls_name}_Precision', 0):.4f} | "
                  f"Recall: {best.get(f'{cls_name}_Recall', 0):.4f}")


def run_training():
    """Main training loop."""
    print(f"\n{'#'*60}")
    print(f"YOLO DETECTION TRAINING FOR DENTAL X-RAY ANALYSIS")
    print(f"{'#'*60}")
    print(f"Run Tag: {RUN_TAG}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Results: {RESULTS_DIR}")

    # Check GPU
    import torch
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available!")
        return

    # GPU configuration
    device = "0,1"  # Using both A5000 GPUs with NVLink

    print("\nGPU Configuration:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"\nUsing GPU(s): {device}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or create master results
    master_file = RESULTS_DIR / f"{RUN_TAG}_master_results.json"
    master_data = load_master_results(master_file)

    # Prepare dataset
    data_yaml = OUTPUT_ROOT / "data.yaml"
    if not data_yaml.exists():
        data_yaml = prepare_yolo_dataset(
            images_root=IMAGES_ROOT,
            masks_root=MASKS_ROOT,
            output_root=OUTPUT_ROOT,
            val_split=0.2,
            seed=42
        )
    else:
        print(f"\nUsing existing dataset: {data_yaml}")

    # Get all models
    models = get_yolo_detection_models()
    total_models = len(models)

    print(f"\nTotal models to train: {total_models}")

    # Train each model
    start_time = datetime.now()
    completed = 0
    failed = 0

    for idx, (model_name, model_config) in enumerate(models.items()):
        print(f"\n[{idx+1}/{total_models}] Processing {model_name}...")

        result = train_single_model(
            model_name=model_name,
            model_config=model_config,
            data_yaml=data_yaml,
            results_dir=RESULTS_DIR,
            device=device,
            train_config=TRAIN_CONFIG,
            master_data=master_data,
            master_file=master_file
        )

        if result:
            completed += 1
        else:
            failed += 1

        # Save progress after each model
        master_data['last_updated'] = datetime.now().isoformat()
        save_master_results(master_file, master_data)

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    master_data['experiment_info']['completed_at'] = end_time.isoformat()
    master_data['experiment_info']['total_duration_hours'] = duration.total_seconds() / 3600
    master_data['experiment_info']['models_completed'] = completed
    master_data['experiment_info']['models_failed'] = failed
    save_master_results(master_file, master_data)

    # Create comparison report
    create_comparison_report(master_data, RESULTS_DIR)

    print(f"\n{'#'*60}")
    print("TRAINING COMPLETED")
    print(f"{'#'*60}")
    print(f"Duration: {duration}")
    print(f"Models trained: {completed}/{total_models}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Master results: {master_file}")


if __name__ == "__main__":
    run_training()
