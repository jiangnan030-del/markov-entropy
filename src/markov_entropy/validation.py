"""Numerical validation helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ATOL = 1e-12


def probability_vector(values: npt.ArrayLike, size: int) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"expected probability vector of shape {(size,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities must be finite")
    if np.any(array < -ATOL):
        raise ValueError("probabilities must be non-negative")
    array = np.maximum(array, 0.0)
    if not np.isclose(array.sum(), 1.0, atol=ATOL, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    array = np.asarray(array / array.sum(), dtype=np.float64)
    array.setflags(write=False)
    return array


def stochastic_matrix(
    values: npt.ArrayLike, rows: int, columns: int
) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, columns):
        raise ValueError(f"expected matrix of shape {(rows, columns)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("channel entries must be finite")
    if np.any(array < -ATOL):
        raise ValueError("channel entries must be non-negative")
    array = np.maximum(array, 0.0)
    sums = array.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=ATOL, rtol=0.0):
        raise ValueError("every channel column must sum to one")
    array = np.asarray(array / sums, dtype=np.float64)
    array.setflags(write=False)
    return array
