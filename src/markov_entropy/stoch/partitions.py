"""Finite-partition lower bounds for divergences on general spaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy.typing as npt

from ..distributions import Distribution
from ..divergences.base import Divergence, distribution_divergence
from ..spaces import FiniteSpace

PartitionLevel = tuple[npt.ArrayLike, npt.ArrayLike]


@dataclass(frozen=True)
class PartitionDivergenceEstimate:
    """Divergence values induced by a sequence of finite partitions."""

    values: tuple[float, ...]
    labels: tuple[str, ...]
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("at least one partition level is required")
        if len(self.values) != len(self.labels):
            raise ValueError("partition values and labels must align")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    @property
    def lower_bound(self) -> float:
        """Largest divergence observed over the supplied partitions."""
        return max(self.values)

    @property
    def is_monotone(self) -> bool:
        """Whether the supplied sequence is non-decreasing within tolerance."""
        return all(
            later + self.tolerance >= earlier
            for earlier, later in zip(self.values, self.values[1:], strict=False)
        )


def partition_lower_bounds(
    levels: Iterable[PartitionLevel],
    divergence: Divergence,
    *,
    labels: Sequence[str] | None = None,
    require_monotone: bool = False,
    tolerance: float = 1e-10,
) -> PartitionDivergenceEstimate:
    """Evaluate divergence after each user-supplied finite partition.

    Each level is a pair of aligned probability-mass vectors. The function
    does not infer or verify that one partition refines another; callers must
    provide that structural guarantee when interpreting monotone convergence.
    """
    level_list = list(levels)
    if not level_list:
        raise ValueError("at least one partition level is required")
    if labels is None:
        label_values = tuple(f"level-{index + 1}" for index in range(len(level_list)))
    else:
        label_values = tuple(labels)
        if len(label_values) != len(level_list):
            raise ValueError("labels must match the number of partition levels")

    values: list[float] = []
    for index, (first_masses, second_masses) in enumerate(level_list):
        first_values = tuple(float(value) for value in first_masses)
        second_values = tuple(float(value) for value in second_masses)
        if len(first_values) != len(second_values):
            raise ValueError(f"partition level {index + 1} is not aligned")
        space = FiniteSpace(range(len(first_values)))
        first = Distribution(space, first_values)
        second = Distribution(space, second_values)
        values.append(distribution_divergence(first, second, divergence))

    estimate = PartitionDivergenceEstimate(tuple(values), label_values, tolerance)
    if require_monotone and not estimate.is_monotone:
        raise ValueError("partition divergence sequence is not monotone")
    return estimate
