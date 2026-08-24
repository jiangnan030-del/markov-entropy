"""Divergence interfaces and shared validation."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..channels import Channel
from ..distributions import Distribution


class Divergence(Protocol):
    def __call__(self, first: npt.ArrayLike, second: npt.ArrayLike) -> float: ...


def aligned_probabilities(
    first: npt.ArrayLike, second: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("probability vectors must be one-dimensional and aligned")
    if np.any(p < 0) or np.any(q < 0) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("probability vectors must sum to one")
    return p, q


def distribution_divergence(
    first: Distribution, second: Distribution, divergence: Divergence
) -> float:
    if first.space != second.space:
        raise ValueError("distributions must be defined on the same space")
    return divergence(first.probabilities, second.probabilities)


def channel_divergence(first: Channel, second: Channel, divergence: Divergence) -> float:
    if first.domain != second.domain or first.codomain != second.codomain:
        raise ValueError("channels must have the same domain and codomain")
    return max(
        divergence(first.matrix[:, index], second.matrix[:, index])
        for index in range(len(first.domain))
    )
