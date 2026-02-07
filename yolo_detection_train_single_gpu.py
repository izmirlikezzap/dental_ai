#!/usr/bin/env python3
"""
Single-GPU YOLO Detection Training Script

Adapted from yolo_detection_train_advanced.py to run on a single GPU.
Designed for GPU job scheduling to enable parallel experiment execution.

Usage:
  python3 yolo_detection_train_single_gpu.py \
    --model yolov8n --model-size n --loss focal --augmented \
    --fold 0 --gpu 0 --epochs 500 --batch-size 24 --image-size 1024
"""

import os
import sys
import csv
import glob
import json
import time
import shutil
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PATHS / DATASET
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path("/home/eva/PycharmProjects/dentalAI/yolo_detection_results_v4")

DATASET = {
    "root": "/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_yolo_detection_v3",
    "images": "/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_yolo_detection_v3/images",
    "labels": "/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_yolo_detection_v3/labels",
}

# Import project modules
from experiment_config import ExperimentManager, create_experiment_config
from custom_losses import get_detection_loss

# -----------------------------------------------------------------------------
# DATASET UTILS
# -----------------------------------------------------------------------------

_cached_all_images: Optional[List[str]] = None

def get_all_images() -> List[str]:
    """Cache dataset images for faster access"""
    global _cached_all_images
    if _cached_all_images is not None:
        return _cached_all_images

    images_dir = Path(DATASET["images"])
    train_images_dir = images_dir / "train"

    paths: List[Path] = []
    if train_images_dir.exists():
        paths.extend(train_images_dir.glob("*.jpg"))
        paths.extend(train_images_dir.glob("*.png"))
        paths.extend(train_images_dir.glob("*.jpeg"))
    else:
        paths.extend(images_dir.glob("**/*.jpg"))
        paths.extend(images_dir.glob("**/*.png"))
        paths.extend(images_dir.glob("**/*.jpeg"))

    uniq = sorted(list({p.resolve() for p in paths}))
    _cached_all_images = [str(p) for p in uniq]
    print(f"[INFO] Cached {len(_cached_all_images)} images from dataset")
    return _cached_all_images


# -----------------------------------------------------------------------------
# TRAINER CLASS
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, config, manager: ExperimentManager, device: str = "0"):
        self.config = config
        self.manager = manager
        self.device = device  # Single GPU device string: "0"
        self.config.device = device

        self.paths = self.manager.create_experiment_structure(
            is_augmented=self.config.is_augmented,
            loss_method=self.config.loss_method,
            model_name=self.config.model_name
        )
        self.config.results_dir = str(self.paths["runs_dir"])
        self.config.weights_dir = str(self.paths["weights_dir"])

    def prepare_data_yaml(self, train_images: List[str], val_images: List[str], fold: int) -> Path:
        fold_dir = self.paths["runs_dir"] / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_txt = fold_dir / "train.txt"
        val_txt = fold_dir / "val.txt"

        train_txt.write_text("\n".join(train_images))
        val_txt.write_text("\n".join(val_images))

        data_yaml = {
            "path": "",
            "train": str(train_txt),
            "val": str(val_txt),
            "nc": 2,
            "names": ["meziodens", "supernumere"]
        }

        yaml_path = fold_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data_yaml, f)

        return yaml_path

    def _parse_class_metrics_from_log(self, run_name: str) -> Dict[str, Dict[str, float]]:
        """
        Parse per-class metrics from training log file.

        Returns:
            Dictionary with class names as keys, each containing:
            {
                'precision': float,
                'recall': float,
                'mAP50': float,
                'mAP50-95': float
            }
        """
        # Try to find the log file in scheduler_logs/job_logs
        log_dir = RESULTS_ROOT / "scheduler_logs" / "job_logs"
        if not log_dir.exists():
            return {}

        # Find log file matching run_name pattern
        # run_name format: yolov5n_default_fold0_20260207_162109
        # log file format: yolov5n_default_noaug_fold0_20260207_162109.log
        # The log file has "noaug" or "aug" inserted, so we need flexible matching

        # Extract model, loss, and fold from run_name
        # Format: {model}_{loss}_fold{N}_{timestamp}
        parts = run_name.split('_')
        if len(parts) < 3:
            return {}

        # Find the fold part
        fold_idx = None
        for i, part in enumerate(parts):
            if part.startswith('fold'):
                fold_idx = i
                break

        if fold_idx is None:
            return {}

        # Build pattern: model_loss_*_fold{N}_*.log
        # This will match both "noaug" and "aug" versions
        model_loss = '_'.join(parts[:fold_idx])  # e.g., "yolov5n_default"
        fold_part = parts[fold_idx]  # e.g., "fold0"
        pattern = str(log_dir / f"{model_loss}_*{fold_part}_*.log")
        log_files = glob.glob(pattern)

        if not log_files:
            return {}

        # Use the most recent log file
        log_file = max(log_files, key=lambda x: Path(x).stat().st_mtime)

        try:
            class_metrics = {}
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Look for class-specific metrics near the end of the file
            # Format: "meziodens         28         36      0.455      0.556      0.593      0.337"
            # Columns: Class, Images, Instances, Precision, Recall, mAP50, mAP50-95
            for line in reversed(lines):
                line = line.strip()
                if 'meziodens' in line.lower():
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            class_metrics['meziodens'] = {
                                'precision': float(parts[-4]),
                                'recall': float(parts[-3]),
                                'mAP50': float(parts[-2]),
                                'mAP50-95': float(parts[-1])
                            }
                        except (ValueError, IndexError):
                            pass
                elif 'supernumere' in line.lower():
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            class_metrics['supernumere'] = {
                                'precision': float(parts[-4]),
                                'recall': float(parts[-3]),
                                'mAP50': float(parts[-2]),
                                'mAP50-95': float(parts[-1])
                            }
                        except (ValueError, IndexError):
                            pass

                # Stop when we have both classes
                if len(class_metrics) == 2:
                    break

            return class_metrics

        except Exception as e:
            print(f"[WARNING] Could not parse class metrics from log: {e}")
            return {}

    def _parse_best_epoch_from_csv(self, results_csv: Path) -> Dict:
        """
        Parse results.csv and extract comprehensive metrics from the best epoch
        (based on mAP50-95 / fitness)

        Returns:
            Dictionary with detection metrics, losses, and per-class metrics
        """
        default_metrics = {
            # Core detection metrics
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fitness": 0.0,
            "f1_score": 0.0,
            # Best epoch info
            "best_epoch": 0,
            "training_time_at_best": 0.0,
            # Training losses (at best epoch)
            "train_box_loss": 0.0,
            "train_cls_loss": 0.0,
            "train_dfl_loss": 0.0,
            # Validation losses (at best epoch)
            "val_box_loss": 0.0,
            "val_cls_loss": 0.0,
            "val_dfl_loss": 0.0,
            # Learning rate (at best epoch)
            "learning_rate": 0.0,
            # Per-class metrics
            "meziodens_AP50": 0.0,
            "meziodens_AP": 0.0,
            "supernumere_AP50": 0.0,
            "supernumere_AP": 0.0,
        }

        if not results_csv.exists():
            print(f"[WARNING] results.csv not found: {results_csv}")
            return default_metrics

        try:
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                print(f"[WARNING] results.csv is empty")
                return default_metrics

            # Find row with best mAP50-95 (fitness)
            best_row = max(rows, key=lambda r: float(r.get('metrics/mAP50-95(B)', 0)))

            # Core metrics
            precision = float(best_row.get('metrics/precision(B)', 0))
            recall = float(best_row.get('metrics/recall(B)', 0))

            # Calculate F1 score
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            metrics = {
                # Core detection metrics
                "mAP50": float(best_row.get('metrics/mAP50(B)', 0)),
                "mAP50-95": float(best_row.get('metrics/mAP50-95(B)', 0)),
                "precision": precision,
                "recall": recall,
                "fitness": float(best_row.get('metrics/mAP50-95(B)', 0)),
                "f1_score": f1_score,
                # Best epoch info
                "best_epoch": int(float(best_row.get('epoch', 0))),
                "training_time_at_best": float(best_row.get('time', 0)),
                # Training losses
                "train_box_loss": float(best_row.get('train/box_loss', 0)),
                "train_cls_loss": float(best_row.get('train/cls_loss', 0)),
                "train_dfl_loss": float(best_row.get('train/dfl_loss', 0)),
                # Validation losses
                "val_box_loss": float(best_row.get('val/box_loss', 0)),
                "val_cls_loss": float(best_row.get('val/cls_loss', 0)),
                "val_dfl_loss": float(best_row.get('val/dfl_loss', 0)),
                # Learning rate
                "learning_rate": float(best_row.get('lr/pg0', 0)),
                # Per-class metrics (will be populated if available)
                "meziodens_AP50": 0.0,
                "meziodens_AP": 0.0,
                "supernumere_AP50": 0.0,
                "supernumere_AP": 0.0,
            }

            return metrics

        except Exception as e:
            print(f"[ERROR] Failed to parse results.csv: {e}")
            return default_metrics

    def train_fold(self, fold: int, train_images: List[str], val_images: List[str]) -> Dict:
        data_yaml = self.prepare_data_yaml(train_images, val_images, fold)

        run_name = self.manager.generate_run_name(
            model_name=self.config.model_name,
            loss_method=self.config.loss_method,
            fold=fold,
            include_timestamp=True
        )

        model = YOLO(self.config.pretrained_weights)

        # Base training arguments
        # Disable built-in early stopping - custom callback will handle it
        train_args = {
            "data": str(data_yaml),
            "epochs": self.config.epochs,
            "patience": 0,  # Disable built-in early stopping, use custom callback instead
            "batch": self.config.batch_size,
            "imgsz": self.config.image_size,
            "device": self.device,  # Single GPU: "0"
            "workers": 4,
            "project": str(self.paths["runs_dir"]),
            "name": run_name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": self.config.learning_rate,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "save": True,
            "save_period": -1,
            "cache": False,
            "plots": False,
            "verbose": True,
        }

        # Fixed loss weights for all methods (actual loss logic is injected via criterion)
        train_args.update({
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
        })
        print(f"[LOSS] Using {self.config.loss_method} loss (injected via on_train_start callback)")

        if self.config.is_augmented:
            train_args.update({
                "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                "degrees": 15.0, "translate": 0.1, "scale": 0.5,
                "shear": 0.0, "perspective": 0.0,
                "flipud": 0.0, "fliplr": 0.5,
                "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
            })
        else:
            train_args.update({
                "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                "shear": 0.0, "perspective": 0.0,
                "flipud": 0.0, "fliplr": 0.0,
                "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
            })

        start_time = time.time()

        # Custom early stopping callback
        # Enforce minimum 50 epochs, then apply patience=20
        class CustomEarlyStopping:
            def __init__(self):
                self.best_fitness = -float('inf')
                self.best_epoch = 0
                self.patience_counter = 0

            def __call__(self, trainer):
                epoch = trainer.epoch

                # Get current fitness (mAP50-95)
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    current_fitness = trainer.metrics.get('fitness', 0.0)
                elif hasattr(trainer, 'best_fitness'):
                    current_fitness = trainer.best_fitness
                else:
                    return

                # Always track best fitness across all epochs
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_epoch = epoch

                    if epoch >= 50:
                        self.patience_counter = 0
                elif epoch >= 50:
                    self.patience_counter += 1

                # Apply early stopping only after epoch 50
                if epoch >= 50 and self.patience_counter >= 20:
                    trainer.stopper.possible_stop = True
                    trainer.stopper.stop = True
                    print(f"\n[CUSTOM EARLY STOP] No improvement for 20 epochs after epoch 50")
                    print(f"  Best epoch: {self.best_epoch} (fitness: {self.best_fitness:.4f})")
                    print(f"  Stopped at epoch: {epoch}")

        early_stopper = CustomEarlyStopping()
        model.add_callback("on_fit_epoch_end", early_stopper)

        # Inject custom loss into model.criterion via on_train_start callback
        loss_method = self.config.loss_method
        loss_params = self.config.loss_params

        def inject_loss(trainer):
            criterion = get_detection_loss(loss_method, trainer.model, loss_params)
            trainer.model.criterion = criterion
            print(f"[LOSS] Injected {loss_method} loss into model.criterion")

        model.add_callback("on_train_start", inject_loss)

        print(f"\n[TRAIN START] {self.config.model_name} | {self.config.loss_method} | "
              f"Fold {fold} | Device: {self.device} | Batch: {self.config.batch_size}")
        print(f"[EARLY STOPPING] Minimum 50 epochs, then patience=20")

        results = model.train(**train_args)

        # Move weights from runs to models directory (don't copy, move to save space)
        weights_source_dir = self.paths["runs_dir"] / run_name / "weights"
        best_weights = weights_source_dir / "best.pt"
        last_weights = weights_source_dir / "last.pt"

        dest_weights: Optional[Path] = None
        if best_weights.exists():
            # Move best.pt to models directory
            dest_weights = self.paths["weights_dir"] / f"{run_name}_best.pt"
            shutil.move(str(best_weights), str(dest_weights))

            # Also move last.pt
            if last_weights.exists():
                dest_last = self.paths["weights_dir"] / f"{run_name}_last.pt"
                shutil.move(str(last_weights), str(dest_last))

            # Remove empty weights directory from runs
            if weights_source_dir.exists():
                try:
                    shutil.rmtree(weights_source_dir)
                except Exception as e:
                    print(f"[WARNING] Could not remove weights dir: {e}")

        training_time = time.time() - start_time

        # Read results.csv to get best epoch metrics
        results_csv = self.paths["runs_dir"] / run_name / "results.csv"
        best_metrics = self._parse_best_epoch_from_csv(results_csv)

        # Parse per-class metrics from log file (if available)
        class_metrics = self._parse_class_metrics_from_log(run_name)
        if class_metrics:
            if 'meziodens' in class_metrics:
                best_metrics['meziodens_AP50'] = class_metrics['meziodens']['mAP50']
                best_metrics['meziodens_AP'] = class_metrics['meziodens']['mAP50-95']
            if 'supernumere' in class_metrics:
                best_metrics['supernumere_AP50'] = class_metrics['supernumere']['mAP50']
                best_metrics['supernumere_AP'] = class_metrics['supernumere']['mAP50-95']

        fold_results = {
            "fold": fold,
            "run_name": run_name,
            "training_time_seconds": training_time,
            "best_weights": str(dest_weights) if dest_weights else None,
            "metrics": best_metrics
        }

        print(f"[TRAIN DONE] {self.config.model_name} | Fold {fold} | "
              f"mAP50: {fold_results['metrics']['mAP50']:.4f} | "
              f"Time: {training_time/60:.1f}min")

        return fold_results


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Single-GPU YOLO Detection Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 yolo_detection_train_single_gpu.py \\
    --model yolov8n --model-size n --loss focal --augmented \\
    --fold 0 --gpu 0 --epochs 500 --batch-size 24 --image-size 1024
        """
    )

    parser.add_argument('--model', required=True,
                       help='Model name (e.g., yolov8n, yolo11m)')
    parser.add_argument('--model-size', required=True,
                       help='Model size code (n, s, m, b, l, x, c, e)')
    parser.add_argument('--loss', required=True,
                       choices=['default', 'focal', 'diou', 'weighted'],
                       help='Loss method')
    parser.add_argument('--augmented', action='store_true',
                       help='Enable YOLO built-in augmentation')
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold number (0-4 for 5-fold CV)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (default: 500)')
    parser.add_argument('--batch-size', type=int, required=True,
                       help='Batch size for training')
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Input image size (default: 1024)')

    args = parser.parse_args()

    # Set CUDA device via environment variable for complete isolation
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set PyTorch multiprocessing sharing strategy
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')

    # After setting CUDA_VISIBLE_DEVICES, the visible GPU is always device "0"
    device = "0"

    # Verify GPU availability
    if not torch.cuda.is_available():
        print(f"[ERROR] CUDA is not available!")
        sys.exit(1)

    print(f"\n{'='*90}")
    print(f"Single-GPU Training Configuration:")
    print(f"  Model: {args.model} ({args.model_size})")
    print(f"  Loss: {args.loss}")
    print(f"  Augmentation: {'Yes' if args.augmented else 'No'}")
    print(f"  Fold: {args.fold}")
    print(f"  Physical GPU: {args.gpu} (mapped to device 0)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Image Size: {args.image_size}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"{'='*90}\n")

    # Build config
    config = create_experiment_config(
        model_name=args.model,
        model_size=args.model_size,
        loss_method=args.loss,
        is_augmented=args.augmented,
        dataset_version="v3",
        n_folds=5,  # K-fold configuration (we'll train one fold)
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
    )

    # Initialize manager and trainer
    manager = ExperimentManager(RESULTS_ROOT)
    trainer = Trainer(config, manager, device=device)

    # Register experiment if not already registered
    if not manager.is_experiment_completed(
        model_name=args.model,
        loss_method=args.loss,
        is_augmented=args.augmented,
        n_folds=5
    ):
        manager.register_experiment(config)

    # Load all images and create k-fold split
    all_images = get_all_images()
    if len(all_images) == 0:
        raise RuntimeError(f"No images found under: {DATASET['images']}")

    # Use same random state as original for consistent splits
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get the specific fold's train/val split
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
        if fold_idx == args.fold:
            train_images = [all_images[i] for i in train_idx]
            val_images = [all_images[i] for i in val_idx]
            break
    else:
        print(f"[ERROR] Invalid fold number: {args.fold}")
        sys.exit(1)

    print(f"[INFO] Training fold {args.fold}: {len(train_images)} train, {len(val_images)} val images")

    try:
        # Train single fold
        fold_results = trainer.train_fold(args.fold, train_images, val_images)

        # Update results in experiment manager
        manager.update_results(
            experiment_id=config.experiment_id,
            fold=args.fold,
            results=fold_results
        )

        # Explicit memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"\n[SUCCESS] Fold {args.fold} completed successfully")
        print(f"  mAP50: {fold_results['metrics']['mAP50']:.4f}")
        print(f"  mAP50-95: {fold_results['metrics']['mAP50-95']:.4f}")
        print(f"  Training time: {fold_results['training_time_seconds']/60:.1f} minutes")

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            print(f"\n[OOM ERROR] CUDA out of memory!")
            print(f"  Model: {args.model}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Suggestion: Reduce batch size and retry")
            print(f"  Error: {str(e)[:200]}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sys.exit(2)  # Special exit code for OOM
        else:
            print(f"\n[RUNTIME ERROR] {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
