#!/usr/bin/env python3
"""
DDP Multi-GPU YOLO Detection Training (2 GPUs with NVLink)

Uses PyTorch DDP (Distributed Data Parallel) to train each model across 2 GPUs.
This provides:
- 48GB total VRAM (24GB + 24GB)
- Faster training through parallelization
- NVLink bandwidth utilization (~56 GB/s)

Models are trained sequentially, each using both GPUs via DDP.

Run:
  python3 yolo_detection_train_advanced.py
"""

import os
import sys
import json
import time
import shutil
import warnings
from pathlib import Path
from datetime import datetime
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

RESULTS_ROOT = Path("/home/eva/PycharmProjects/dentalAI/yolo_detection_results_v3")

DATASET = {
    "root": "/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_yolo_detection_v3",
    "images": "/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_yolo_detection_v3/images",
    "labels": "/mnt/storage_fast_8tb/datasets/jaw_dataset_rana/yolo_format_yolo_detection_v3/labels",
}

# Import project modules
from experiment_config import ExperimentManager, create_experiment_config
from custom_losses import get_loss_function

# -----------------------------------------------------------------------------
# EXPERIMENT SPACE
# -----------------------------------------------------------------------------

MODELS = [
    # YOLOv5
    ("yolov5n", "n"), ("yolov5s", "s"), ("yolov5m", "m"), ("yolov5l", "l"), ("yolov5x", "x"),
    # YOLOv8
    ("yolov8n", "n"), ("yolov8s", "s"), ("yolov8m", "m"), ("yolov8l", "l"), ("yolov8x", "x"),
    # YOLOv9
    ("yolov9c", "c"), ("yolov9e", "e"),
    # YOLOv10
    ("yolov10n", "n"), ("yolov10s", "s"), ("yolov10m", "m"), ("yolov10b", "b"),
    ("yolov10l", "l"), ("yolov10x", "x"),
    # YOLOv11
    ("yolo11n", "n"), ("yolo11s", "s"), ("yolo11m", "m"), ("yolo11l", "l"), ("yolo11x", "x"),
    # YOLO26
    ("yolo26n", "n"), ("yolo26s", "s"), ("yolo26m", "m"), ("yolo26l", "l"), ("yolo26x", "x"),

]

LOSS_METHODS = ["default", "focal", "diou", "weighted"]
AUGMENTATION_SETTINGS = [False, True]   # Phase 1: False, Phase 2: True

N_FOLDS = 5
EPOCHS = 500
IMAGE_SIZE = 1024


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def get_optimal_batch_size(model_size: str) -> int:
    """
    Get optimal batch size for DDP training (2 GPUs).
    Batch size is per-GPU in DDP, so effective batch size = batch_size * num_gpus

    Reduced batch sizes to prevent OOM with 1024x1024 images and limited workers.
    """
    batch_size_map = {
        "n": 24,  # Reduced from 48
        "s": 16,  # Reduced from 32
        "m": 12,  # Reduced from 32
        "b": 10,  # Reduced from 24
        "l": 8,   # Reduced from 16
        "x": 6,   # Reduced from 12
        "c": 10,  # Reduced from 24
        "e": 6,   # Reduced from 14
    }
    return batch_size_map.get(model_size, 8)


def build_all_experiments() -> List[Dict]:
    exps: List[Dict] = []
    for model_name, model_size in MODELS:
        for loss_method in LOSS_METHODS:
            for is_aug in AUGMENTATION_SETTINGS:
                exps.append({
                    "model_name": model_name,
                    "model_size": model_size,
                    "loss_method": loss_method,
                    "is_augmented": is_aug,
                    "n_folds": N_FOLDS,
                    "epochs": EPOCHS,
                    "image_size": IMAGE_SIZE,
                })
    return exps


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
    def __init__(self, config, manager: ExperimentManager, device: str = "0,1"):
        self.config = config
        self.manager = manager
        self.device = device  # DDP device string: "0,1"
        self.config.device = device

        self.paths = self.manager.create_experiment_structure(
            is_augmented=self.config.is_augmented,
            loss_method=self.config.loss_method,
            model_name=self.config.model_name
        )
        self.config.results_dir = str(self.paths["runs_dir"])
        self.config.weights_dir = str(self.paths["weights_dir"])

        # Get loss function (metadata only, not directly injected)
        _ = get_loss_function(self.config.loss_method, self.config.loss_params)

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

    def train_fold(self, fold: int, train_images: List[str], val_images: List[str]) -> Dict:
        data_yaml = self.prepare_data_yaml(train_images, val_images, fold)

        run_name = self.manager.generate_run_name(
            model_name=self.config.model_name,
            loss_method=self.config.loss_method,
            fold=fold,
            include_timestamp=True
        )

        model = YOLO(self.config.pretrained_weights)

        train_args = {
            "data": str(data_yaml),
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "batch": self.config.batch_size,
            "imgsz": self.config.image_size,
            "device": self.device,  # DDP: "0,1" for 2 GPUs
            "workers": 4,  # Reduced from 8 to prevent OOM with 1024x1024 images
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
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "save": True,
            "save_period": -1,
            "cache": False,  # Keep False to prevent memory issues
            "plots": False,
            "verbose": True,
        }

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

        print(f"\n[TRAIN START] {self.config.model_name} | {self.config.loss_method} | "
              f"Fold {fold} | Device: {self.device} | Batch: {self.config.batch_size}")

        results = model.train(**train_args)

        best_weights = self.paths["runs_dir"] / run_name / "weights" / "best.pt"
        dest_weights: Optional[Path] = None
        if best_weights.exists():
            dest_weights = self.paths["weights_dir"] / f"{run_name}_best.pt"
            shutil.copy2(best_weights, dest_weights)

            # Load best.pt and perform final validation with it
            best_model = YOLO(best_weights)
            val_results = best_model.val(data=str(data_yaml), device=self.device)
        else:
            # Fallback: use last weights (shouldn't happen)
            val_results = model.val(data=str(data_yaml), device=self.device)

        training_time = time.time() - start_time

        fold_results = {
            "fold": fold,
            "run_name": run_name,
            "training_time_seconds": training_time,
            "best_weights": str(dest_weights) if dest_weights else None,
            "metrics": {
                "mAP50": float(val_results.box.map50),
                "mAP50-95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr),
                "fitness": float(val_results.fitness),
            }
        }

        # Per-class AP if available
        if hasattr(val_results.box, "ap50") and hasattr(val_results.box, "ap"):
            names = getattr(self.config, "class_names", ["meziodens", "supernumere"])
            for i, cname in enumerate(names):
                if i < len(val_results.box.ap50):
                    fold_results["metrics"][f"{cname}_AP50"] = float(val_results.box.ap50[i])
                    fold_results["metrics"][f"{cname}_AP"] = float(val_results.box.ap[i])

        print(f"[TRAIN DONE] {self.config.model_name} | Fold {fold} | "
              f"mAP50: {fold_results['metrics']['mAP50']:.4f} | "
              f"Time: {training_time/60:.1f}min")

        return fold_results

    def run_kfold(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.manager.register_experiment(self.config)

        all_images = get_all_images()
        if len(all_images) == 0:
            raise RuntimeError(f"No images found under: {DATASET['images']}")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        all_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_images)):
            train_images = [all_images[i] for i in train_idx]
            val_images = [all_images[i] for i in val_idx]

            fold_results = self.train_fold(fold, train_images, val_images)

            self.manager.update_results(
                experiment_id=self.config.experiment_id,
                fold=fold,
                results=fold_results
            )
            all_results.append(fold_results)

        # Save config for traceability
        config_file = self.paths["exp_base"] / f"{self.config.experiment_id}_config.json"
        self.config.to_json(config_file)

        # Calculate and print average metrics
        avg_map50 = np.mean([r["metrics"]["mAP50"] for r in all_results])
        avg_map = np.mean([r["metrics"]["mAP50-95"] for r in all_results])
        print(f"\n[EXPERIMENT COMPLETE] {self.config.model_name} | {self.config.loss_method}")
        print(f"  Average mAP50: {avg_map50:.4f}")
        print(f"  Average mAP50-95: {avg_map:.4f}")


# -----------------------------------------------------------------------------
# EXPERIMENT RUNNER
# -----------------------------------------------------------------------------

def run_single_experiment(exp: Dict, manager: ExperimentManager, oom_log_file: Path) -> None:
    """Run a single experiment with DDP on 2 GPUs"""

    aug_str = "Aug" if exp["is_augmented"] else "No-Aug"

    # Skip if completed
    if manager.is_experiment_completed(
        model_name=exp["model_name"],
        loss_method=exp["loss_method"],
        is_augmented=exp["is_augmented"],
        n_folds=exp["n_folds"]
    ):
        print(f"[SKIP] {exp['model_name']} | {exp['loss_method']} | {aug_str} (already completed)")
        return

    optimal_bs = get_optimal_batch_size(exp["model_size"])

    print(f"\n{'='*90}")
    print(f"[RUN] {exp['model_name']} | {exp['loss_method']} | {aug_str} | BS={optimal_bs} | Device: 0,1 (DDP)")
    print(f"{'='*90}")

    try:
        # Build config
        config = create_experiment_config(
            model_name=exp["model_name"],
            model_size=exp["model_size"],
            loss_method=exp["loss_method"],
            is_augmented=exp["is_augmented"],
            dataset_version="v3",
            n_folds=exp["n_folds"],
            epochs=exp["epochs"],
            batch_size=optimal_bs,
            image_size=exp["image_size"],
            device="0,1",  # DDP
        )

        trainer = Trainer(config, manager, device="0,1")
        trainer.run_kfold(n_splits=exp["n_folds"])

        # Explicit memory cleanup after each experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"[SUCCESS] {exp['model_name']} | {exp['loss_method']} | {aug_str}")

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            # Log OOM error
            oom_errors = []
            if oom_log_file.exists():
                try:
                    oom_errors = json.loads(oom_log_file.read_text())
                except Exception:
                    pass

            record = {
                "timestamp": datetime.now().isoformat(),
                "model_name": exp["model_name"],
                "model_size": exp["model_size"],
                "loss_method": exp["loss_method"],
                "is_augmented": exp["is_augmented"],
                "batch_size": optimal_bs,
                "image_size": exp["image_size"],
                "device": "0,1 (DDP)",
                "error_message": str(e)[:300],
                "suggested_batch_size": max(2, optimal_bs - 2),  # Reduce by 2, minimum 2
            }
            oom_errors.append(record)
            try:
                oom_log_file.write_text(json.dumps(oom_errors, indent=2))
            except Exception:
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[OOM ERROR] {exp['model_name']} | {exp['loss_method']} | {aug_str}")
            print(f"  Suggestion: Reduce batch size to {max(2, optimal_bs - 2)}")
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[RUNTIME ERROR] {exp['model_name']}: {str(e)}")

    except Exception as e:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[ERROR] {exp['model_name']}: {str(e)}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    # Setup for DDP - DO NOT mask GPUs, DDP needs to see all GPUs
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Memory optimization settings to prevent OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set PyTorch multiprocessing sharing strategy
    # 'file_system' is more stable than 'file_descriptor' for large datasets
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')

    # Verify GPU availability
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*90}")
    print(f"GPU Configuration:")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"{'='*90}\n")

    if num_gpus < 2:
        print("[WARNING] Less than 2 GPUs detected. DDP works best with 2+ GPUs.")
        print(f"[INFO] Proceeding with {num_gpus} GPU(s)...")

    # Build experiments
    all_exps = build_all_experiments()

    # Strict phase split
    phase_no_aug = [e for e in all_exps if not e["is_augmented"]]
    phase_aug = [e for e in all_exps if e["is_augmented"]]

    print("=" * 90)
    print("DDP TRAINING PLAN (Sequential with 2-GPU DDP)")
    print(f"Total experiments: {len(all_exps)}")
    print(f"Phase 1 (No-Aug): {len(phase_no_aug)}")
    print(f"Phase 2 (Aug):    {len(phase_aug)}")
    print("=" * 90)

    # Initialize manager and OOM log
    manager = ExperimentManager(RESULTS_ROOT)
    oom_log_file = RESULTS_ROOT / "oom_errors_ddp.json"

    # Run Phase 1: No augmentation
    print(f"\n{'='*90}")
    print(f"PHASE 1: NO-AUG (normal dataset)")
    print(f"{'='*90}\n")

    for exp in phase_no_aug:
        run_single_experiment(exp, manager, oom_log_file)

    # Run Phase 2: Augmentation
    print(f"\n{'='*90}")
    print(f"PHASE 2: AUG (YOLO built-in augmentation)")
    print(f"{'='*90}\n")

    for exp in phase_aug:
        run_single_experiment(exp, manager, oom_log_file)

    print("\n" + "=" * 90)
    print("ALL TRAINING COMPLETED")
    print("=" * 90)
    print("View results with:")
    print("  python manage_experiments.py summary")
    print("  python manage_experiments.py export all_results.csv")


if __name__ == "__main__":
    main()
