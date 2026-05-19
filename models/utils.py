"""Shared utility functions for all models."""

import numpy as np


def random_int(min_val: int, max_val: int) -> int:
    """
    Generate random integer in range [min_val, max_val].
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Returns:
        Random integer within range
    """
    return np.random.randint(min_val, max_val + 1)


def random_float(min_val: float, max_val: float) -> float:
    """
    Generate random float in range [min_val, max_val].
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Random float within range
    """
    return np.random.uniform(min_val, max_val)


def random_log_uniform(min_val: float, max_val: float) -> float:
    """
    Generate random float in log-uniform range [min_val, max_val].
    Useful for hyperparameters like learning rate that span multiple orders of magnitude.
    
    Args:
        min_val: Minimum value (must be > 0)
        max_val: Maximum value
        
    Returns:
        Random float in log-uniform distribution
    """
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    return np.exp(np.random.uniform(log_min, log_max))


def random_batch_size(min_val: int, max_val: int) -> int:
    """
    Generate random batch size (power of 2) in range [min_val, max_val].
    Powers of 2 are typically optimal for GPU memory alignment.
    
    Args:
        min_val: Minimum value (should be power of 2)
        max_val: Maximum value (should be power of 2)
        
    Returns:
        Random batch size that is a power of 2
    """
    powers = np.arange(int(np.log2(min_val)), int(np.log2(max_val)) + 1)
    return 2 ** np.random.choice(powers)
