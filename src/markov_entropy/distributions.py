"""Probability distributions in FinStoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .spaces import FiniteSpace
from .validation import probability_vector


@dataclass(frozen=True, eq=False)
class Distribution:
    """A probability source on a finite space."""

    space: FiniteSpace
    probabilities: npt.NDArray[np.float64]

    def __init__(self, space: FiniteSpace, probabilities: npt.ArrayLike) -> None:
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "probabilities", probability_vector(probabilities, len(space)))

    @classmethod
    def product(cls, *distributions: "Distribution") -> "Distribution":
        if not distributions:
            raise ValueError("at least one distribution is required")
        result = distributions[0]
        for distribution in distributions[1:]:
            result = result.tensor(distribution)
        return result

    def tensor(self, other: "Distribution") -> "Distribution":
        return Distribution(
            self.space.tensor(other.space),
            np.kron(self.probabilities, other.probabilities),
        )

    def marginal(self, axes: int | Iterable[int]) -> "Distribution":
        selected = (axes,) if isinstance(axes, int) else tuple(axes)
        factors = self.space.factors
        if not selected:
            raise ValueError("at least one marginal axis is required")
        if len(set(selected)) != len(selected):
            raise ValueError("marginal axes must be unique")
        if any(axis < 0 or axis >= len(factors) for axis in selected):
            raise ValueError("marginal axis out of range")
        tensor = self.probabilities.reshape(self.space.shape)
        summed = tuple(axis for axis in range(len(factors)) if axis not in selected)
        values = tensor.sum(axis=summed) if summed else tensor
        remaining_order = tuple(axis for axis in range(len(factors)) if axis in selected)
        if selected != remaining_order and len(selected) > 1:
            permutation = tuple(remaining_order.index(axis) for axis in selected)
            values = np.transpose(values, permutation)
        space = factors[selected[0]]
        for axis in selected[1:]:
            space = space.tensor(factors[axis])
        return Distribution(space, np.asarray(values).reshape(-1))

    def __matmul__(self, other: "Distribution") -> "Distribution":
        return self.tensor(other)
