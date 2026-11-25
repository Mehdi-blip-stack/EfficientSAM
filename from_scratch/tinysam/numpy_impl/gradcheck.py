"""Central finite-difference tools for checking handwritten derivatives."""

import numpy as np


def finite_difference(array, scalar_function, eps=1e-5, indices=None):
    """Estimate selected entries of ∂f/∂array using centered differences."""
    numerical = np.zeros_like(array)
    indices = list(np.ndindex(array.shape)) if indices is None else list(indices)
    for index in indices:
        original = array[index]
        array[index] = original + eps
        plus = scalar_function()
        array[index] = original - eps
        minus = scalar_function()
        array[index] = original
        numerical[index] = (plus - minus) / (2 * eps)
    return numerical


def relative_error(analytic, numeric):
    """Elementwise symmetric relative error with a stable zero denominator."""
    return np.abs(analytic - numeric) / np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)


def sampled_indices(shape, count=8, seed=0):
    """Pick deterministic distinct coordinates from a tensor."""
    if not shape:
        return [()]
    rng = np.random.default_rng(seed)
    total = int(np.prod(shape))
    flat = rng.choice(total, size=min(count, total), replace=False)
    return [np.unravel_index(i, shape) for i in flat]
