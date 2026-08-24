"""Rényi divergences on finite distributions."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .base import aligned_probabilities
from .kl import KL


class Renyi:
    """Rényi divergence of order ``alpha`` using natural logarithms."""

    def __init__(self, alpha: float) -> None:
        if math.isnan(alpha) or alpha < 0:
            raise ValueError("alpha must lie in [0, infinity]")
        self.alpha = float(alpha)

    def __call__(self, first: npt.ArrayLike, second: npt.ArrayLike) -> float:
        p, q = aligned_probabilities(first, second)
        alpha = self.alpha
        if alpha == 1.0:
            return KL()(p, q)
        if alpha == 0.0:
            mass = float(q[p > 0].sum())
            return -math.log(mass) if mass > 0 else float("inf")
        if math.isinf(alpha):
            if np.any((p > 0) & (q == 0)):
                return float("inf")
            mask = p > 0
            return float(np.log(np.max(p[mask] / q[mask])))
        if alpha > 1 and np.any((p > 0) & (q == 0)):
            return float("inf")
        mask = (p > 0) & (q > 0)
        total = float(np.sum(np.power(p[mask], alpha) * np.power(q[mask], 1 - alpha)))
        if total == 0:
            return float("inf")
        return math.log(total) / (alpha - 1)
