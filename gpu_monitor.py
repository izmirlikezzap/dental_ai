#!/usr/bin/env python3
"""
GPU Monitoring Module
Provides utilities for checking GPU availability and memory usage.
"""

import subprocess
import time
import logging
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


def get_gpu_memory_info(gpu_id: int) -> Dict[str, float]:
    """
    Get memory information for a specific GPU using nvidia-smi.

    Args:
        gpu_id: GPU index (0, 1, etc.)

    Returns:
        Dictionary with keys: free_mb, total_mb, used_mb, utilization
        Returns None values if query fails
    """
    try:
        result = subprocess.run([
            'nvidia-smi',
            f'--id={gpu_id}',
            '--query-gpu=memory.free,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            logger.error(f"nvidia-smi failed for GPU {gpu_id}: {result.stderr}")
            return {'free_mb': None, 'total_mb': None, 'used_mb': None, 'utilization': None}

        parts = result.stdout.strip().split(',')
        free_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        utilization = float(parts[2].strip())
        used_mb = total_mb - free_mb

        return {
            'free_mb': free_mb,
            'total_mb': total_mb,
            'used_mb': used_mb,
            'utilization': utilization
        }

    except subprocess.TimeoutExpired:
        logger.error(f"nvidia-smi timeout for GPU {gpu_id}")
        return {'free_mb': None, 'total_mb': None, 'used_mb': None, 'utilization': None}
    except Exception as e:
        logger.error(f"Error querying GPU {gpu_id}: {e}")
        return {'free_mb': None, 'total_mb': None, 'used_mb': None, 'utilization': None}


def is_gpu_available(gpu_id: int, required_memory_gb: float, utilization_threshold: float = 90.0) -> bool:
    """
    Check if a GPU meets availability requirements.

    Args:
        gpu_id: GPU index to check
        required_memory_gb: Required free memory in GB
        utilization_threshold: Maximum acceptable GPU utilization percentage

    Returns:
        True if GPU is available, False otherwise
    """
    info = get_gpu_memory_info(gpu_id)

    if info['free_mb'] is None or info['utilization'] is None:
        return False

    required_memory_mb = required_memory_gb * 1024
    free_enough = info['free_mb'] >= required_memory_mb
    not_busy = info['utilization'] < utilization_threshold

    logger.debug(
        f"GPU {gpu_id}: {info['free_mb']:.0f} MB free / {info['total_mb']:.0f} MB total "
        f"(Util: {info['utilization']:.0f}%) - Required: {required_memory_mb:.0f} MB, "
        f"Available: {free_enough and not_busy}"
    )

    return free_enough and not_busy


def get_available_gpu(gpu_ids: List[int], required_memory_gb: float,
                     utilization_threshold: float = 90.0) -> Optional[int]:
    """
    Find the first available GPU from a list.

    Args:
        gpu_ids: List of GPU indices to check
        required_memory_gb: Required free memory in GB
        utilization_threshold: Maximum acceptable GPU utilization percentage

    Returns:
        First available GPU ID, or None if none available
    """
    for gpu_id in gpu_ids:
        if is_gpu_available(gpu_id, required_memory_gb, utilization_threshold):
            return gpu_id
    return None


def wait_for_available_gpu(gpu_ids: List[int], required_memory_gb: float,
                          timeout: int = 3600, poll_interval: int = 30,
                          utilization_threshold: float = 90.0) -> Optional[int]:
    """
    Block until a GPU becomes available or timeout is reached.

    Args:
        gpu_ids: List of GPU indices to monitor
        required_memory_gb: Required free memory in GB
        timeout: Maximum wait time in seconds (default: 1 hour)
        poll_interval: Time between checks in seconds (default: 30s)
        utilization_threshold: Maximum acceptable GPU utilization percentage

    Returns:
        Available GPU ID, or None if timeout reached
    """
    start_time = time.time()
    last_log_time = 0

    while True:
        # Check for available GPU
        gpu_id = get_available_gpu(gpu_ids, required_memory_gb, utilization_threshold)
        if gpu_id is not None:
            logger.info(f"GPU {gpu_id} became available")
            return gpu_id

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            logger.warning(f"Timeout waiting for GPU after {elapsed:.0f} seconds")
            return None

        # Log status periodically (every 5 minutes)
        if elapsed - last_log_time >= 300:
            logger.info(f"Still waiting for GPU... ({elapsed:.0f}s elapsed)")
            for gid in gpu_ids:
                info = get_gpu_memory_info(gid)
                if info['free_mb'] is not None:
                    logger.info(
                        f"  GPU {gid}: {info['free_mb']:.0f} MB free, "
                        f"{info['utilization']:.0f}% utilized"
                    )
            last_log_time = elapsed

        # Sleep before next check
        time.sleep(poll_interval)


def get_all_gpus_info(gpu_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Get memory information for all specified GPUs.

    Args:
        gpu_ids: List of GPU indices to query

    Returns:
        Dictionary mapping GPU ID to memory info dict
    """
    return {gpu_id: get_gpu_memory_info(gpu_id) for gpu_id in gpu_ids}


if __name__ == "__main__":
    # Test GPU monitoring
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("GPU Monitoring Test")
    print("=" * 60)

    # Check which GPUs are available
    available_gpus = []
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index',
            '--format=csv,noheader'
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            available_gpus = [int(idx.strip()) for idx in result.stdout.strip().split('\n')]
            print(f"Detected GPUs: {available_gpus}\n")
        else:
            print("Failed to detect GPUs")
            sys.exit(1)
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        sys.exit(1)

    # Display info for all GPUs
    all_info = get_all_gpus_info(available_gpus)
    for gpu_id, info in all_info.items():
        if info['free_mb'] is not None:
            print(f"GPU {gpu_id}:")
            print(f"  Total Memory: {info['total_mb']:.0f} MB ({info['total_mb']/1024:.1f} GB)")
            print(f"  Used Memory:  {info['used_mb']:.0f} MB ({info['used_mb']/1024:.1f} GB)")
            print(f"  Free Memory:  {info['free_mb']:.0f} MB ({info['free_mb']/1024:.1f} GB)")
            print(f"  Utilization:  {info['utilization']:.0f}%")
            print()

    # Test availability checks
    test_requirements = [4, 8, 12, 16]
    print("\nAvailability Check (90% utilization threshold):")
    print("-" * 60)
    for req_gb in test_requirements:
        print(f"Required: {req_gb} GB")
        for gpu_id in available_gpus:
            available = is_gpu_available(gpu_id, req_gb, 90.0)
            status = "✓ Available" if available else "✗ Not Available"
            print(f"  GPU {gpu_id}: {status}")
        print()
