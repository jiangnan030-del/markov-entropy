"""Kullback-Leibler divergence."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .base import aligned_probabilities


class KL:
    """Natural-log KL divergence ``D(p || q)``."""

    def __call__(self, first: npt.ArrayLike, second: npt.ArrayLike) -> float:
        p, q = aligned_probabilities(first, second)
        if np.any((p > 0) & (q == 0)):
            return float("inf")
        mask = p > 0
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
