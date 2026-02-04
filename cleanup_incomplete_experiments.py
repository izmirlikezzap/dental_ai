#!/usr/bin/env python3
"""
Cleanup incomplete experiments and identify what needs to be trained

This script:
1. Identifies completed experiments (all 5 folds done)
2. Identifies incomplete experiments (partial folds)
3. Removes incomplete experiment data
4. Reports what needs to be trained
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from experiment_config import ExperimentManager, LOSS_CONFIGS

RESULTS_ROOT = Path("/home/eva/PycharmProjects/dentalAI/yolo_detection_results_v3")
EXPECTED_FOLDS = 5

def analyze_experiments(manager: ExperimentManager) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Analyze experiments and categorize them

    Returns:
        (completed, incomplete, not_started)
    """
    master = manager._load_json(manager.master_results_file)
    tracker = manager._load_json(manager.experiment_tracker_file)

    completed = []
    incomplete = []

    for exp in tracker.get('experiments', []):
        exp_id = exp['id']

        if exp_id not in master:
            # No results at all - will be in not_started
            continue

        exp_data = master[exp_id]
        folds_data = exp_data.get('folds', {})
        num_folds = len(folds_data)

        if num_folds >= EXPECTED_FOLDS:
            # Check if weights exist
            paths = manager.create_experiment_structure(
                is_augmented=exp['is_augmented'],
                loss_method=exp['loss_method'],
                model_name=exp['model']
            )

            weights_dir = paths['weights_dir']
            weight_files = list(weights_dir.glob(f"{exp['model']}_{exp['loss_method']}_fold*_best.pt"))

            if len(weight_files) >= EXPECTED_FOLDS:
                completed.append(exp)
            else:
                incomplete.append({
                    'exp': exp,
                    'reason': f'Missing weights: {len(weight_files)}/{EXPECTED_FOLDS}',
                    'folds_completed': num_folds
                })
        else:
            incomplete.append({
                'exp': exp,
                'reason': f'Incomplete folds: {num_folds}/{EXPECTED_FOLDS}',
                'folds_completed': num_folds
            })

    return completed, incomplete


def cleanup_incomplete_experiment(exp_info: Dict, manager: ExperimentManager) -> None:
    """Remove all traces of an incomplete experiment"""
    exp = exp_info['exp']
    exp_id = exp['id']

    print(f"  Cleaning: {exp['model']} | {exp['loss_method']} | "
          f"{'Aug' if exp['is_augmented'] else 'No-Aug'} | "
          f"Reason: {exp_info['reason']}")

    # Remove from master_results.json
    master = manager._load_json(manager.master_results_file)
    if exp_id in master:
        del master[exp_id]
        manager._save_json(master, manager.master_results_file)

    # Remove from experiment_tracker.json
    tracker = manager._load_json(manager.experiment_tracker_file)
    tracker['experiments'] = [e for e in tracker.get('experiments', []) if e['id'] != exp_id]
    manager._save_json(tracker, manager.experiment_tracker_file)

    # Remove experiment directories
    paths = manager.create_experiment_structure(
        is_augmented=exp['is_augmented'],
        loss_method=exp['loss_method'],
        model_name=exp['model']
    )

    exp_base = paths['exp_base']
    if exp_base.exists():
        print(f"    Removing directory: {exp_base}")
        shutil.rmtree(exp_base, ignore_errors=True)


def main():
    print("=" * 90)
    print("EXPERIMENT CLEANUP AND ANALYSIS")
    print("=" * 90)

    manager = ExperimentManager(RESULTS_ROOT)

    # Analyze experiments
    completed, incomplete = analyze_experiments(manager)

    print(f"\nCompleted experiments: {len(completed)}")
    print(f"Incomplete experiments: {len(incomplete)}")

    # Show completed experiments
    if completed:
        print("\n" + "=" * 90)
        print("COMPLETED EXPERIMENTS (will be skipped during training):")
        print("=" * 90)
        for exp in completed:
            aug_str = "Aug" if exp['is_augmented'] else "No-Aug"
            print(f"  ✓ {exp['model']:15s} | {exp['loss_method']:10s} | {aug_str}")

    # Show incomplete experiments
    if incomplete:
        print("\n" + "=" * 90)
        print("INCOMPLETE EXPERIMENTS (will be cleaned and retrained):")
        print("=" * 90)
        for exp_info in incomplete:
            exp = exp_info['exp']
            aug_str = "Aug" if exp['is_augmented'] else "No-Aug"
            print(f"  ⚠ {exp['model']:15s} | {exp['loss_method']:10s} | {aug_str:6s} | {exp_info['reason']}")

    # Auto-cleanup incomplete experiments
    if incomplete:
        print("\n" + "=" * 90)
        print(f"Auto-cleaning {len(incomplete)} incomplete experiments...")
        print("=" * 90)
        for exp_info in incomplete:
            cleanup_incomplete_experiment(exp_info, manager)
        print(f"\n✓ Cleaned {len(incomplete)} incomplete experiments.")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY:")
    print("=" * 90)
    print(f"Completed: {len(completed)} experiments")
    print(f"Incomplete (cleaned): {len(incomplete) if incomplete else 0} experiments")
    print(f"\nReady to train remaining experiments.")
    print("=" * 90)


if __name__ == "__main__":
    main()
