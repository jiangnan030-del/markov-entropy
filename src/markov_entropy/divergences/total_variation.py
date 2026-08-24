"""Total variation distance."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .base import aligned_probabilities


class TotalVariation:
    def __call__(self, first: npt.ArrayLike, second: npt.ArrayLike) -> float:
        p, q = aligned_probabilities(first, second)
        return float(0.5 * np.abs(p - q).sum())
