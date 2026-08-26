"""Rényi divergences on finite distributions."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .base import aligned_probabilities
from .kl import KL


class Renyi:
    """Rényi divergence of order ``alpha`` using natural logarithms.

    Finite orders are evaluated in the log domain. Close to ``alpha = 1``,
    ``log1p`` and ``expm1`` avoid the cancellation present in the direct
    power-sum formula.
    """

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
            return float(np.max(np.log(p[mask]) - np.log(q[mask])))
        if alpha > 1 and np.any((p > 0) & (q == 0)):
            return float("inf")

        mask = (p > 0) & (q > 0)
        if not np.any(mask):
            return float("inf")

        log_p = np.log(p[mask])
        log_q = np.log(q[mask])
        delta = alpha - 1.0
        log_ratio = log_p - log_q

        support_is_contained = bool(np.all((p == 0) | (q > 0)))
        if abs(delta) < 1e-6 and support_is_contained:
            correction = float(np.sum(p[mask] * np.expm1(delta * log_ratio)))
            log_total = math.log1p(correction)
        else:
            log_terms = log_p + delta * log_ratio
            log_total = float(np.logaddexp.reduce(log_terms))
        return log_total / delta
