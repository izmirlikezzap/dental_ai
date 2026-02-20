#!/usr/bin/env python3
"""
GPU Job Scheduler for YOLO Detection Training

Automatically schedules and executes training jobs across available GPUs,
maximizing resource utilization and enabling parallel experiment execution.

Usage:
  python3 gpu_job_scheduler.py --gpus 0,1
  python3 gpu_job_scheduler.py --gpus 0,1 --dry-run
"""

import os
import sys
import json
import csv
import time
import signal
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from queue import PriorityQueue
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_monitor import get_available_gpu, get_all_gpus_info, is_gpu_available
from experiment_config import ExperimentManager

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

RESULTS_ROOT = Path("/home/eva/PycharmProjects/dentalAI/yolo_detection_results_v4")
LOG_DIR = RESULTS_ROOT / "scheduler_logs"
JOB_LOG_DIR = LOG_DIR / "job_logs"

# Model configurations - All sizes
MODELS = [
    # YOLOv5
    ("yolov5n", "n"),
    ("yolov5s", "s"), ("yolov5m", "m"), ("yolov5l", "l"), ("yolov5x", "x"),

    # YOLOv8
    ("yolov8n", "n"),
    ("yolov8s", "s"), ("yolov8m", "m"), ("yolov8l", "l"), ("yolov8x", "x"),

    # YOLOv9
    ("yolov9c", "c"),
    ("yolov9e", "e"),

    # YOLOv10
    ("yolov10n", "n"),
    ("yolov10s", "s"), ("yolov10m", "m"), ("yolov10b", "b"),
    ("yolov10l", "l"), ("yolov10x", "x"),

    # YOLOv11
    ("yolo11n", "n"),
    ("yolo11s", "s"), ("yolo11m", "m"), ("yolo11l", "l"), ("yolo11x", "x"),

    # YOLO26
    ("yolo26n", "n"),
    ("yolo26s", "s"), ("yolo26m", "m"), ("yolo26l", "l"), ("yolo26x", "x"),
]

LOSS_METHODS = ["default", "focal", "diou", "weighted"]
AUGMENTATION_SETTINGS = [False, True]
N_FOLDS = 5
EPOCHS = 500
IMAGE_SIZE = 1024

# Batch sizes optimized for RTX A5000 (24GB VRAM) with 1024x1024 images
BATCH_SIZE_MAP = {
    "n": 48,  # Nano models - aggressive start, OOM retry will reduce by 4
    "s": 32,  # Small models
    "m": 24,  # Medium models
    "b": 16,  # Balanced models (YOLOv10)
    "l": 12,  # Large models
    "x": 10,  # Extra large models
    "c": 12,  # Compact models (YOLOv9c) - 20 OOMs on A5000
    "e": 10,  # Efficient models (YOLOv9e)
}

# Required memory for each model size (GB)
MEMORY_REQUIREMENTS = {
    "n": 4, "s": 6, "m": 8, "b": 10,
    "l": 12, "x": 16, "c": 12, "e": 16,
}

# Job timeout disabled (no time limit)
JOB_TIMEOUT = None

# GPU polling intervals
POLL_INTERVAL_ACTIVE = 5  # Check every 5 seconds when jobs are running
POLL_INTERVAL_WAITING = 30  # Check every 30 seconds when waiting for GPU
POLL_INTERVAL_BACKOFF = 60  # Check every 60 seconds after prolonged waiting

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to both file and console"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('scheduler')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# -----------------------------------------------------------------------------
# JOB CLASS
# -----------------------------------------------------------------------------

@dataclass(order=True)
class Job:
    # Priority field (lower = higher priority)
    priority: int = field(init=False, repr=False)

    # Job configuration
    model_name: str = field(compare=False)
    model_size: str = field(compare=False)
    loss_method: str = field(compare=False)
    is_augmented: bool = field(compare=False)
    fold: int = field(compare=False)
    batch_size: int = field(compare=False)
    epochs: int = field(default=EPOCHS, compare=False)
    image_size: int = field(default=IMAGE_SIZE, compare=False)
    retry_count: int = field(default=0, compare=False)

    def __post_init__(self):
        self.priority = self.calculate_priority()

    def calculate_priority(self) -> int:
        """
        Calculate job priority (lower = higher priority).

        Priority system:
        - Phase 1 (no augmentation) before Phase 2 (augmentation)
        - Smaller models before larger models
        - Natural fold ordering
        """
        # Phase weight: no-aug = 0, aug = 10000
        phase_weight = 0 if not self.is_augmented else 10000

        # Model size weight (n=0, s=100, m=200, etc.)
        size_order = {"n": 0, "s": 100, "m": 200, "b": 300, "l": 400, "x": 500, "c": 300, "e": 500}
        model_weight = size_order.get(self.model_size, 300)

        # Fold weight (0-4)
        fold_weight = self.fold

        return phase_weight + model_weight + fold_weight

    def get_required_memory_gb(self) -> float:
        """Get required GPU memory in GB"""
        return MEMORY_REQUIREMENTS.get(self.model_size, 8)

    def get_job_id(self) -> str:
        """Get unique job identifier"""
        aug_str = "aug" if self.is_augmented else "noaug"
        return f"{self.model_name}_{self.loss_method}_{aug_str}_fold{self.fold}"

    def to_command(self, gpu_id: int) -> str:
        """Build subprocess command"""
        aug_flag = "--augmented" if self.is_augmented else ""
        cmd = (
            f"python3 yolo_detection_train_single_gpu.py "
            f"--model {self.model_name} "
            f"--model-size {self.model_size} "
            f"--loss {self.loss_method} "
            f"{aug_flag} "
            f"--fold {self.fold} "
            f"--gpu {gpu_id} "
            f"--epochs {self.epochs} "
            f"--batch-size {self.batch_size} "
            f"--image-size {self.image_size}"
        )
        return cmd.strip()


# -----------------------------------------------------------------------------
# SCHEDULER CLASS
# -----------------------------------------------------------------------------

class Scheduler:
    def __init__(self, gpu_ids: List[int], results_dir: Path, logger: logging.Logger):
        self.gpu_ids = gpu_ids
        self.results_dir = results_dir
        self.logger = logger
        self.manager = ExperimentManager(results_dir)

        # Active jobs: {gpu_id: job_info_dict}
        self.active_jobs: Dict[int, Dict] = {}

        # Completed and failed jobs
        self.completed_jobs: List[Dict] = []
        self.failed_jobs: List[Dict] = []

        # Scheduler state
        self.should_stop = False
        self.start_time = datetime.now()

        # Summary file
        self.summary_file = LOG_DIR / "scheduler_summary.json"

        # CSV files for each loss method
        self.csv_files = {}
        self._initialize_csv_files()

        # Progress bar (will be set during schedule_loop)
        self.pbar = None

    def _initialize_csv_files(self):
        """Initialize CSV files for each loss method"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for loss_method in LOSS_METHODS:
            csv_file = LOG_DIR / f"results_{loss_method}_{timestamp}.csv"
            self.csv_files[loss_method] = csv_file

            # Create CSV with headers
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'job_id',
                    'model_name',
                    'model_size',
                    'loss_method',
                    'augmentation',
                    'fold',
                    'batch_size',
                    'gpu_id',
                    'start_time',
                    'end_time',
                    'duration_minutes',
                    'status',
                    'exit_code',
                    'retry_count',
                    'log_file'
                ])

            self.logger.info(f"Created CSV file for {loss_method}: {csv_file}")

    def _log_to_csv(self, record: Dict):
        """Log job result to the appropriate CSV file"""
        loss_method = record['loss_method']
        csv_file = self.csv_files.get(loss_method)

        if not csv_file:
            self.logger.error(f"No CSV file found for loss method: {loss_method}")
            return

        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record['end_time'],
                    record['job_id'],
                    record['model_name'],
                    record['model_size'],
                    record['loss_method'],
                    'Yes' if record['is_augmented'] else 'No',
                    record['fold'],
                    record['batch_size'],
                    record['gpu_id'],
                    record['start_time'],
                    record['end_time'],
                    round(record['duration_seconds'] / 60, 2),
                    record['status'],
                    record['exit_code'],
                    record['retry_count'],
                    record['log_file']
                ])
        except Exception as e:
            self.logger.error(f"Failed to write to CSV {csv_file}: {e}")

    def _is_fold_completed(self, model_name: str, loss_method: str,
                          is_augmented: bool, fold: int) -> bool:
        """
        Check if a specific fold of an experiment is completed.

        Args:
            model_name: Model name (e.g., 'yolov8n')
            loss_method: Loss method (e.g., 'focal')
            is_augmented: Augmentation status
            fold: Fold number (0-4)

        Returns:
            True if fold is completed, False otherwise
        """
        # Check weight file existence
        paths = self.manager.create_experiment_structure(
            is_augmented=is_augmented,
            loss_method=loss_method,
            model_name=model_name
        )

        weights_dir = paths['weights_dir']
        if not weights_dir.exists():
            return False

        # Look for weight files matching this fold
        # Pattern: {model}_{loss}_fold{n}_*_best.pt
        weight_pattern = f"{model_name}_{loss_method}_fold{fold}_*_best.pt"
        weight_files = list(weights_dir.glob(weight_pattern))

        return len(weight_files) > 0

    def build_job_queue(self) -> PriorityQueue:
        """Build priority queue of all pending jobs"""
        self.logger.info("Building job queue...")

        job_queue = PriorityQueue()
        total_jobs = 0
        skipped_jobs = 0

        for model_name, model_size in MODELS:
            for loss_method in LOSS_METHODS:
                for is_augmented in AUGMENTATION_SETTINGS:
                    # Check if experiment is completed (all folds done)
                    if self.manager.is_experiment_completed(
                        model_name=model_name,
                        loss_method=loss_method,
                        is_augmented=is_augmented,
                        n_folds=N_FOLDS
                    ):
                        self.logger.info(
                            f"[SKIP] {model_name} | {loss_method} | "
                            f"{'Aug' if is_augmented else 'No-Aug'} (completed)"
                        )
                        skipped_jobs += N_FOLDS
                        continue

                    # Check individual folds
                    for fold in range(N_FOLDS):
                        # Check if this specific fold is completed
                        fold_completed = self._is_fold_completed(
                            model_name=model_name,
                            loss_method=loss_method,
                            is_augmented=is_augmented,
                            fold=fold
                        )

                        if fold_completed:
                            self.logger.debug(
                                f"[SKIP] {model_name} | {loss_method} | "
                                f"{'Aug' if is_augmented else 'No-Aug'} | Fold {fold} (completed)"
                            )
                            skipped_jobs += 1
                            continue

                        # Create job
                        batch_size = BATCH_SIZE_MAP.get(model_size, 8)
                        job = Job(
                            model_name=model_name,
                            model_size=model_size,
                            loss_method=loss_method,
                            is_augmented=is_augmented,
                            fold=fold,
                            batch_size=batch_size,
                            epochs=EPOCHS,
                            image_size=IMAGE_SIZE,
                        )
                        job_queue.put(job)
                        total_jobs += 1

        self.logger.info(f"Job queue built: {total_jobs} jobs pending, {skipped_jobs} jobs skipped")
        return job_queue

    def _validate_pt_file(self, pt_file: str) -> bool:
        """Check if a .pt file is valid by attempting to load it with torch."""
        try:
            import torch
            torch.load(pt_file, map_location="cpu", weights_only=False)
            return True
        except Exception:
            return False

    def download_pretrained_weights(self, job_queue: PriorityQueue, redownload: bool = False):
        """Pre-download all required .pt files to avoid race conditions.

        When multiple GPUs try to download the same .pt file simultaneously,
        one process may read a partially downloaded/corrupt file, causing
        SIGABRT (exit code -6) from torch.load().

        Also validates existing .pt files and re-downloads corrupt ones.
        """
        unique_models = set()
        for job in job_queue.queue:
            unique_models.add(f"{job.model_name}.pt")

        if not unique_models:
            return

        self.logger.info(f"Checking {len(unique_models)} model weight files...")
        for pt_file in sorted(unique_models):
            pt_path = Path(pt_file)
            if pt_path.exists() and not redownload:
                # Validate existing file
                if self._validate_pt_file(pt_file):
                    self.logger.info(f"  {pt_file} OK")
                    continue
                else:
                    self.logger.warning(f"  {pt_file} is CORRUPT, deleting and re-downloading...")
                    pt_path.unlink()
            elif pt_path.exists() and redownload:
                self.logger.info(f"  {pt_file} exists, --redownload set, deleting...")
                pt_path.unlink()

            self.logger.info(f"  Downloading {pt_file}...")
            try:
                from ultralytics import YOLO
                YOLO(pt_file)  # triggers download
                if self._validate_pt_file(pt_file):
                    self.logger.info(f"  Downloaded and verified {pt_file}")
                else:
                    self.logger.error(f"  Downloaded {pt_file} but validation failed!")
            except Exception as e:
                self.logger.error(f"  Failed to download {pt_file}: {e}")

    def launch_job(self, job: Job, gpu_id: int) -> bool:
        """Launch training subprocess on specified GPU"""
        if gpu_id in self.active_jobs:
            self.logger.error(f"GPU {gpu_id} already has an active job!")
            return False

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = JOB_LOG_DIR / f"{job.get_job_id()}_{timestamp}.log"

        # Build command
        cmd = job.to_command(gpu_id)

        # Prepare environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        try:
            # Launch subprocess
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=PROJECT_ROOT
                )

            # Track active job
            self.active_jobs[gpu_id] = {
                'job': job,
                'process': process,
                'start_time': datetime.now(),
                'log_file': log_file,
                'pid': process.pid
            }

            self.logger.info(
                f"[JOB LAUNCH] {job.get_job_id()} on GPU {gpu_id} "
                f"(PID: {process.pid}, Batch: {job.batch_size})"
            )
            return True

        except Exception as e:
            self.logger.error(f"[JOB LAUNCH FAILED] {job.get_job_id()}: {e}")
            return False

    def check_active_jobs(self) -> List[Tuple[int, int]]:
        """
        Check for completed jobs.

        Returns:
            List of (gpu_id, returncode) tuples for completed jobs
        """
        completed = []
        for gpu_id, job_info in list(self.active_jobs.items()):
            process = job_info['process']
            returncode = process.poll()

            if returncode is not None:
                # Job completed
                completed.append((gpu_id, returncode))
            else:
                # Check for timeout (disabled if JOB_TIMEOUT is None)
                if JOB_TIMEOUT is not None:
                    elapsed = (datetime.now() - job_info['start_time']).total_seconds()
                    if elapsed > JOB_TIMEOUT:
                        self.logger.warning(
                            f"[JOB TIMEOUT] {job_info['job'].get_job_id()} on GPU {gpu_id} "
                            f"(exceeded {JOB_TIMEOUT/3600:.1f} hours)"
                        )
                        process.kill()
                        completed.append((gpu_id, -1))

        return completed

    def handle_job_completion(self, gpu_id: int, returncode: int, job_queue: PriorityQueue):
        """Process completed job"""
        job_info = self.active_jobs.pop(gpu_id)
        job = job_info['job']
        duration = (datetime.now() - job_info['start_time']).total_seconds()

        record = {
            "job_id": job.get_job_id(),
            "model_name": job.model_name,
            "model_size": job.model_size,
            "loss_method": job.loss_method,
            "is_augmented": job.is_augmented,
            "fold": job.fold,
            "batch_size": job.batch_size,
            "gpu_id": gpu_id,
            "start_time": job_info['start_time'].isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "exit_code": returncode,
            "log_file": str(job_info['log_file']),
            "retry_count": job.retry_count,
        }

        if returncode == 0:
            # Success
            record["status"] = "SUCCESS"
            self.completed_jobs.append(record)
            self._log_to_csv(record)  # Log to CSV
            self.logger.info(
                f"[JOB COMPLETE] {job.get_job_id()} on GPU {gpu_id} "
                f"(Duration: {duration/60:.1f}min, Status: SUCCESS)"
            )

        elif returncode == 2:
            # OOM error (exit code 2 from training script)
            record["status"] = "OOM"

            # Check retry limit
            if job.retry_count < 4:
                # Reduce batch size and retry
                new_batch_size = max(2, job.batch_size - 8)
                self.logger.warning(
                    f"[JOB OOM] {job.get_job_id()} on GPU {gpu_id} "
                    f"(Retry {job.retry_count + 1}/4, reducing batch size {job.batch_size} -> {new_batch_size})"
                )

                # Log OOM retry to CSV
                self._log_to_csv(record)

                # Create retry job
                retry_job = Job(
                    model_name=job.model_name,
                    model_size=job.model_size,
                    loss_method=job.loss_method,
                    is_augmented=job.is_augmented,
                    fold=job.fold,
                    batch_size=new_batch_size,
                    epochs=job.epochs,
                    image_size=job.image_size,
                    retry_count=job.retry_count + 1,
                )
                job_queue.put(retry_job)
            else:
                # Max retries exceeded
                self.logger.error(
                    f"[JOB FAILED] {job.get_job_id()} on GPU {gpu_id} "
                    f"(OOM, max retries exceeded)"
                )
                self.failed_jobs.append(record)
                self._log_to_csv(record)  # Log to CSV

        else:
            # Other failure
            record["status"] = "FAILED"
            self.failed_jobs.append(record)
            self._log_to_csv(record)  # Log to CSV
            self.logger.error(
                f"[JOB FAILED] {job.get_job_id()} on GPU {gpu_id} "
                f"(Exit code: {returncode})"
            )

    def save_summary(self):
        """Save scheduler summary to JSON"""
        summary = {
            "scheduler_start": self.start_time.isoformat(),
            "scheduler_end": datetime.now().isoformat(),
            "total_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "statistics": {
                "total_jobs": len(self.completed_jobs) + len(self.failed_jobs),
                "completed": len(self.completed_jobs),
                "failed": len(self.failed_jobs),
            },
            "jobs": self.completed_jobs + self.failed_jobs,
        }

        # Add per-GPU statistics
        for gpu_id in self.gpu_ids:
            gpu_jobs = [j for j in summary["jobs"] if j["gpu_id"] == gpu_id]
            summary["statistics"][f"gpu_{gpu_id}_jobs"] = len(gpu_jobs)

        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Summary saved to {self.summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")

    def schedule_loop(self, job_queue: PriorityQueue):
        """Main scheduling loop"""
        self.logger.info(f"Starting scheduler loop with GPUs: {self.gpu_ids}")

        # Display initial GPU status
        gpu_info = get_all_gpus_info(self.gpu_ids)
        for gpu_id, info in gpu_info.items():
            if info['free_mb'] is not None:
                self.logger.info(
                    f"[GPU] GPU {gpu_id}: {info['free_mb']:.0f} MB free / "
                    f"{info['total_mb']:.0f} MB total (Util: {info['utilization']:.0f}%)"
                )

        # Calculate total jobs for progress bar
        total_jobs = job_queue.qsize()

        # Temporarily disable console handler to avoid interference with tqdm
        console_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                console_handler = handler
                self.logger.removeHandler(handler)
                break

        # Create progress bar
        pbar = tqdm(
            total=total_jobs,
            desc="Training Progress",
            unit="job",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            position=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout
        )

        # Store pbar in self for access from other methods
        self.pbar = pbar

        waiting_start_time = None

        while not job_queue.empty() or self.active_jobs:
            if self.should_stop:
                self.logger.warning("Stop signal received, waiting for active jobs to complete...")
                break

            # Check for completed jobs
            completed = self.check_active_jobs()
            for gpu_id, returncode in completed:
                self.handle_job_completion(gpu_id, returncode, job_queue)
                pbar.update(1)  # Update progress bar

            # Update progress bar description with GPU status
            gpu_status = []
            for gpu_id in self.gpu_ids:
                if gpu_id in self.active_jobs:
                    job = self.active_jobs[gpu_id]['job']
                    elapsed = (datetime.now() - self.active_jobs[gpu_id]['start_time']).total_seconds() / 60
                    gpu_status.append(f"GPU{gpu_id}:{job.model_name[:8]}({elapsed:.0f}m)")
                else:
                    gpu_status.append(f"GPU{gpu_id}:idle")

            pbar.set_description(f"Training [{', '.join(gpu_status)}]")

            # Try to launch new jobs on available GPUs
            if not job_queue.empty():
                for gpu_id in self.gpu_ids:
                    if gpu_id in self.active_jobs:
                        continue  # GPU busy

                    if job_queue.empty():
                        break

                    # Get next job
                    job = job_queue.get()

                    # Check GPU availability
                    required_memory = job.get_required_memory_gb()
                    if not is_gpu_available(gpu_id, required_memory, utilization_threshold=90.0):
                        # GPU not ready, put job back
                        job_queue.put(job)
                        self.logger.debug(
                            f"[GPU] GPU {gpu_id} not available for {job.get_job_id()} "
                            f"(requires {required_memory} GB)"
                        )
                        continue

                    # Launch job
                    if self.launch_job(job, gpu_id):
                        waiting_start_time = None  # Reset waiting timer
                    else:
                        # Launch failed, put job back
                        job_queue.put(job)

            # Determine polling interval
            if self.active_jobs:
                # Jobs running, check frequently
                poll_interval = POLL_INTERVAL_ACTIVE
            elif not job_queue.empty():
                # Waiting for GPU
                if waiting_start_time is None:
                    waiting_start_time = time.time()
                    poll_interval = POLL_INTERVAL_WAITING
                elif time.time() - waiting_start_time > 300:  # 5 minutes
                    poll_interval = POLL_INTERVAL_BACKOFF
                else:
                    poll_interval = POLL_INTERVAL_WAITING
            else:
                break

            time.sleep(poll_interval)

        # Wait for remaining active jobs
        if self.active_jobs:
            self.logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete...")
            timeout = 300  # 5 minutes
            start_wait = time.time()

            while self.active_jobs and (time.time() - start_wait) < timeout:
                completed = self.check_active_jobs()
                for gpu_id, returncode in completed:
                    self.handle_job_completion(gpu_id, returncode, job_queue)
                    pbar.update(1)  # Update progress bar
                time.sleep(5)

            # Kill remaining jobs if timeout
            if self.active_jobs:
                self.logger.warning("Timeout waiting for jobs, killing remaining processes...")
                for gpu_id, job_info in self.active_jobs.items():
                    try:
                        job_info['process'].kill()
                        self.logger.warning(f"Killed job on GPU {gpu_id}: {job_info['job'].get_job_id()}")
                    except Exception as e:
                        self.logger.error(f"Failed to kill job on GPU {gpu_id}: {e}")

        # Close progress bar
        pbar.close()
        self.pbar = None

        # Restore console handler
        if console_handler is not None:
            self.logger.addHandler(console_handler)

        # Save summary
        self.save_summary()

        self.logger.info("=" * 90)
        self.logger.info("SCHEDULER COMPLETED")
        self.logger.info(f"Total jobs completed: {len(self.completed_jobs)}")
        self.logger.info(f"Total jobs failed: {len(self.failed_jobs)}")
        self.logger.info(f"Total duration: {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} hours")
        self.logger.info("")
        self.logger.info("CSV files by loss method:")
        for loss_method, csv_file in self.csv_files.items():
            self.logger.info(f"  {loss_method}: {csv_file}")
        self.logger.info("=" * 90)

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GPU Job Scheduler for YOLO Detection Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on GPUs 0 and 1
  python3 gpu_job_scheduler.py --gpus 0,1

  # Dry run (show job queue without executing)
  python3 gpu_job_scheduler.py --gpus 0,1 --dry-run

  # Run on single GPU
  python3 gpu_job_scheduler.py --gpus 0
        """
    )

    parser.add_argument('--gpus', default='0,1',
                       help='Comma-separated list of GPU IDs to use (default: 0,1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Build job queue and show plan without executing')
    parser.add_argument('--redownload', action='store_true',
                       help='Delete and re-download all pretrained .pt weight files')

    args = parser.parse_args()

    # Parse GPU IDs
    try:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    except ValueError:
        print("[ERROR] Invalid GPU IDs. Use comma-separated integers (e.g., '0,1')")
        sys.exit(1)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"scheduler_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 90)
    logger.info("GPU JOB SCHEDULER")
    logger.info("=" * 90)
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Results directory: {RESULTS_ROOT}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 90)

    # Verify GPU availability
    gpu_info = get_all_gpus_info(gpu_ids)
    for gpu_id in gpu_ids:
        if gpu_info[gpu_id]['free_mb'] is None:
            logger.error(f"GPU {gpu_id} is not accessible!")
            sys.exit(1)

    # Create scheduler
    scheduler = Scheduler(gpu_ids, RESULTS_ROOT, logger)

    # Setup signal handlers
    signal.signal(signal.SIGINT, scheduler.signal_handler)
    signal.signal(signal.SIGTERM, scheduler.signal_handler)

    # Build job queue
    job_queue = scheduler.build_job_queue()

    # Pre-download pretrained weights to avoid race conditions between GPUs
    if not args.dry_run:
        scheduler.download_pretrained_weights(job_queue, redownload=args.redownload)

    if args.dry_run:
        logger.info("\n" + "=" * 90)
        logger.info("DRY RUN - Job Queue Preview")
        logger.info("=" * 90)

        # Display first 20 jobs
        preview_jobs = []
        while not job_queue.empty() and len(preview_jobs) < 20:
            job = job_queue.get()
            preview_jobs.append(job)

        for i, job in enumerate(preview_jobs, 1):
            aug_str = "Aug" if job.is_augmented else "No-Aug"
            logger.info(
                f"{i:3d}. {job.model_name:10s} | {job.loss_method:8s} | {aug_str:6s} | "
                f"Fold {job.fold} | BS={job.batch_size:2d} | Mem={job.get_required_memory_gb():2.0f}GB | "
                f"Priority={job.priority}"
            )

        if not job_queue.empty():
            logger.info(f"... and {job_queue.qsize()} more jobs")

        logger.info("\n[DRY RUN] Exiting without execution")
        return

    # Run scheduler
    scheduler.schedule_loop(job_queue)

    logger.info("\nView results at:")
    logger.info(f"  JSON Summary: {scheduler.summary_file}")
    logger.info(f"  CSV files (by loss method):")
    for loss_method, csv_file in scheduler.csv_files.items():
        logger.info(f"    {loss_method}: {csv_file}")


if __name__ == "__main__":
    main()
