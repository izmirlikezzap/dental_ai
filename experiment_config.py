#!/usr/bin/env python3
"""
Experiment Configuration and Tracking System
Manages different loss methods, augmentation settings, and experiment metadata
Thread-safe with file locking for parallel GPU training
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
import yaml
import fcntl
import time


# Loss method configurations
LOSS_CONFIGS = {
    'default': {
        'name': 'default_loss',
        'description': 'YOLO v8 default loss (box + cls + dfl)',
        'params': {}
    },
    'focal': {
        'name': 'focal_loss',
        'description': 'Focal Loss for hard example mining',
        'params': {
            'alpha': 0.25,
            'gamma': 2.0
        }
    },
    'diou': {
        'name': 'diou_loss',
        'description': 'Distance-IoU Loss for better bbox regression',
        'params': {
            'loss_type': 'diou'  # or 'ciou'
        }
    },
    'weighted': {
        'name': 'weighted_loss',
        'description': 'Class-weighted loss for imbalanced classes',
        'params': {
            'class_weights': [1.0, 2.0, 2.0]  # [healthy, meziodens, supernumere]
        }
    }
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    # Experiment identifiers
    experiment_id: str
    experiment_name: str
    timestamp: str

    # Model configuration
    model_name: str
    model_size: str
    pretrained_weights: str

    # Data configuration
    dataset_version: str
    is_augmented: bool
    augmentation_types: List[str]
    num_classes: int
    class_names: List[str]

    # Training configuration
    loss_method: str
    loss_params: Dict[str, Any]
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    patience: int

    # K-Fold configuration
    n_folds: int
    current_fold: Optional[int] = None

    # GPU configuration
    device: str = 'cuda:0'

    # Paths
    results_dir: Optional[str] = None
    weights_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, filepath: Path):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ExperimentManager:
    """Manages experiment directory structure and metadata"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Models directory (for weights storage, separate from results)
        # Place it alongside the base_dir with "_models" suffix
        base_parent = self.base_dir.parent
        base_name = self.base_dir.name
        self.models_dir = base_parent / f"{base_name}_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Master tracking files
        self.master_results_file = self.base_dir / "master_results.json"
        self.experiment_tracker_file = self.base_dir / "experiment_tracker.json"

        # Initialize tracking files if they don't exist
        if not self.master_results_file.exists():
            self._save_json({}, self.master_results_file)

        if not self.experiment_tracker_file.exists():
            self._save_json({"experiments": []}, self.experiment_tracker_file)

    def create_experiment_structure(
        self,
        is_augmented: bool,
        loss_method: str,
        model_name: str,
        fold: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        Create directory structure for an experiment

        Returns:
            Dictionary with paths: runs_dir, weights_dir, metadata_file
        """
        # Determine base path
        aug_dir = "augmented" if is_augmented else "normal"
        loss_name = LOSS_CONFIGS[loss_method]['name']

        # Create directory hierarchy for results
        exp_base = self.base_dir / aug_dir / loss_name
        runs_dir = exp_base / "runs"

        # Create directory hierarchy for weights (in separate models directory)
        models_base = self.models_dir / aug_dir / loss_name
        weights_dir = models_base / "weights"

        runs_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file for this specific loss/aug combination
        metadata_file = exp_base / "experiment_metadata.json"

        return {
            'exp_base': exp_base,
            'runs_dir': runs_dir,
            'weights_dir': weights_dir,
            'metadata_file': metadata_file
        }

    def generate_run_name(
        self,
        model_name: str,
        loss_method: str,
        fold: Optional[int] = None,
        include_timestamp: bool = True
    ) -> str:
        """
        Generate run directory name

        Format: {model}_{loss_short}_fold{n}_{timestamp}
        Example: yolov10n_focal_fold0_20260118_143022
        """
        loss_short = loss_method  # 'default', 'focal', 'diou', 'weighted'

        parts = [model_name, loss_short]

        if fold is not None:
            parts.append(f"fold{fold}")

        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)

        return "_".join(parts)

    def register_experiment(self, config: ExperimentConfig):
        """Register a new experiment in the tracker"""
        tracker = self._load_json(self.experiment_tracker_file)

        experiment_entry = {
            'id': config.experiment_id,
            'name': config.experiment_name,
            'timestamp': config.timestamp,
            'model': config.model_name,
            'loss_method': config.loss_method,
            'is_augmented': config.is_augmented,
            'dataset_version': config.dataset_version,
            'n_folds': config.n_folds,
            'config': config.to_dict()
        }

        tracker['experiments'].append(experiment_entry)
        self._save_json(tracker, self.experiment_tracker_file)

    def update_results(
        self,
        experiment_id: str,
        fold: int,
        results: Dict[str, Any]
    ):
        """Update results for a specific experiment and fold"""
        master = self._load_json(self.master_results_file)

        if experiment_id not in master:
            master[experiment_id] = {
                'folds': {},
                'avg_metrics': {}
            }

        master[experiment_id]['folds'][f'fold_{fold}'] = results

        # Calculate average metrics across folds
        self._calculate_avg_metrics(master[experiment_id])

        self._save_json(master, self.master_results_file)

    def _calculate_avg_metrics(self, experiment_data: Dict):
        """Calculate average metrics across all folds"""
        folds_data = experiment_data['folds']

        if not folds_data:
            return

        # Collect all metrics
        metrics_sum = {}
        metrics_count = {}

        for fold_results in folds_data.values():
            for metric, value in fold_results.items():
                if isinstance(value, (int, float)):
                    if metric not in metrics_sum:
                        metrics_sum[metric] = 0
                        metrics_count[metric] = 0
                    metrics_sum[metric] += value
                    metrics_count[metric] += 1

        # Calculate averages
        avg_metrics = {
            metric: metrics_sum[metric] / metrics_count[metric]
            for metric in metrics_sum
        }

        experiment_data['avg_metrics'] = avg_metrics

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        tracker = self._load_json(self.experiment_tracker_file)
        master = self._load_json(self.master_results_file)

        summary = {
            'total_experiments': len(tracker['experiments']),
            'by_augmentation': {
                'augmented': 0,
                'normal': 0
            },
            'by_loss_method': {},
            'experiments': []
        }

        for exp in tracker['experiments']:
            # Count by augmentation
            if exp['is_augmented']:
                summary['by_augmentation']['augmented'] += 1
            else:
                summary['by_augmentation']['normal'] += 1

            # Count by loss method
            loss = exp['loss_method']
            summary['by_loss_method'][loss] = summary['by_loss_method'].get(loss, 0) + 1

            # Add experiment info with results
            exp_id = exp['id']
            exp_summary = exp.copy()
            if exp_id in master:
                exp_summary['results'] = master[exp_id]['avg_metrics']

            summary['experiments'].append(exp_summary)

        return summary

    def list_experiments(
        self,
        loss_method: Optional[str] = None,
        is_augmented: Optional[bool] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        tracker = self._load_json(self.experiment_tracker_file)
        experiments = tracker['experiments']

        # Apply filters
        if loss_method is not None:
            experiments = [e for e in experiments if e['loss_method'] == loss_method]

        if is_augmented is not None:
            experiments = [e for e in experiments if e['is_augmented'] == is_augmented]

        if model_name is not None:
            experiments = [e for e in experiments if e['model'] == model_name]

        return experiments

    def is_experiment_completed(
        self,
        model_name: str,
        loss_method: str,
        is_augmented: bool,
        n_folds: int = 5
    ) -> bool:
        """
        Check if an experiment is already completed

        Args:
            model_name: Model name (e.g., 'yolov8n')
            loss_method: Loss method (e.g., 'focal')
            is_augmented: Augmentation status
            n_folds: Expected number of folds

        Returns:
            True if experiment is completed, False otherwise
        """
        # Get matching experiments
        matching = self.list_experiments(
            loss_method=loss_method,
            is_augmented=is_augmented,
            model_name=model_name
        )

        if not matching:
            return False

        # Load master results once
        master = self._load_json(self.master_results_file)

        # Check ALL matching experiments - if ANY is completed, return True
        for experiment in matching:
            exp_id = experiment['id']

            # Check if results exist
            if exp_id not in master:
                continue

            # Check if all folds are completed
            folds_data = master[exp_id].get('folds', {})
            completed_folds = len(folds_data)

            if completed_folds >= n_folds:
                # Found a completed experiment
                return True

        # No completed experiments found, check weights as final validation
        paths = self.create_experiment_structure(
            is_augmented=is_augmented,
            loss_method=loss_method,
            model_name=model_name
        )

        weights_dir = paths['weights_dir']
        if not weights_dir.exists():
            return False

        # Count weight files for this model
        weight_files = list(weights_dir.glob(f"{model_name}_{loss_method}_fold*_best.pt"))

        return len(weight_files) >= n_folds

    @staticmethod
    def _load_json(filepath: Path) -> Dict:
        """Load JSON file with retry logic for parallel access"""
        max_retries = 10
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                with open(filepath, 'r') as f:
                    # Acquire shared lock for reading
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    # Return empty dict on final failure
                    return {} if isinstance(e, json.JSONDecodeError) else {"experiments": []}

        return {}

    @staticmethod
    def _save_json(data: Dict, filepath: Path):
        """Save JSON file with exclusive lock for parallel access"""
        max_retries = 10
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Create parent directory if needed
                filepath.parent.mkdir(parents=True, exist_ok=True)

                # Open file in read-write mode (create if doesn't exist)
                # DO NOT truncate before acquiring lock!
                with open(filepath, 'a+') as f:
                    # Acquire exclusive lock FIRST
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        # NOW truncate the file after lock is acquired
                        f.seek(0)
                        f.truncate()
                        # Write the data
                        json.dump(data, f, indent=2)
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise


def create_experiment_config(
    model_name: str,
    model_size: str,
    loss_method: str,
    is_augmented: bool,
    dataset_version: str = "v3",
    n_folds: int = 5,
    epochs: int = 500,
    batch_size: int = 12,
    image_size: int = 1024,
    learning_rate: float = 0.001,
    patience: int = 20,
    device: str = 'cuda:0'
) -> ExperimentConfig:
    """
    Create experiment configuration

    Args:
        model_name: e.g., 'yolov10n', 'yolov8m'
        model_size: e.g., 'n', 's', 'm', 'l', 'x'
        loss_method: 'default', 'focal', 'diou', 'weighted'
        is_augmented: True if using augmented dataset
        dataset_version: Dataset version identifier
        n_folds: Number of k-folds
        epochs: Training epochs
        batch_size: Batch size
        image_size: Input image size
        learning_rate: Learning rate
        patience: Early stopping patience
        device: CUDA device

    Returns:
        ExperimentConfig object
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate experiment ID
    aug_str = "aug" if is_augmented else "normal"
    experiment_id = f"{model_name}_{loss_method}_{aug_str}_{timestamp}"

    # Experiment name
    experiment_name = f"{model_name} with {loss_method} loss ({aug_str})"

    # Augmentation types (if augmented)
    augmentation_types = []
    if is_augmented:
        augmentation_types = [
            'rotation_plus_15',
            'rotation_minus_15',
            'clahe',
            'negative_invert'
        ]

    # Get loss parameters
    loss_config = LOSS_CONFIGS[loss_method]

    # Class names (from your dataset)
    class_names = ['healthy', 'meziodens', 'supernumere']

    return ExperimentConfig(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        timestamp=timestamp,
        model_name=model_name,
        model_size=model_size,
        pretrained_weights=f"{model_name}.pt",
        dataset_version=dataset_version,
        is_augmented=is_augmented,
        augmentation_types=augmentation_types,
        num_classes=len(class_names),
        class_names=class_names,
        loss_method=loss_method,
        loss_params=loss_config['params'],
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        patience=patience,
        n_folds=n_folds,
        device=device
    )


# Example usage
if __name__ == "__main__":
    # Initialize experiment manager
    manager = ExperimentManager(Path("/home/eva/PycharmProjects/dentalAI/yolo_detection_results_v3"))

    # Create experiment config
    config = create_experiment_config(
        model_name="yolov10n",
        model_size="n",
        loss_method="focal",
        is_augmented=True,
        n_folds=5
    )

    # Create directory structure
    paths = manager.create_experiment_structure(
        is_augmented=config.is_augmented,
        loss_method=config.loss_method,
        model_name=config.model_name
    )

    print(f"Experiment ID: {config.experiment_id}")
    print(f"Runs directory: {paths['runs_dir']}")
    print(f"Weights directory: {paths['weights_dir']}")

    # Register experiment
    manager.register_experiment(config)

    # Get summary
    summary = manager.get_experiment_summary()
    print(json.dumps(summary, indent=2))
