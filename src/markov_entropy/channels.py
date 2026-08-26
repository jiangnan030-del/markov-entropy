"""Column-stochastic channels in FinStoch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .distributions import Distribution
from .spaces import FiniteSpace
from .validation import ATOL, stochastic_matrix


@dataclass(frozen=True, eq=False)
class Channel:
    """A channel ``domain -> codomain`` represented by a column-stochastic matrix."""

    domain: FiniteSpace
    codomain: FiniteSpace
    matrix: npt.NDArray[np.float64]

    def __init__(
        self, domain: FiniteSpace, codomain: FiniteSpace, matrix: npt.ArrayLike
    ) -> None:
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codomain", codomain)
        object.__setattr__(
            self, "matrix", stochastic_matrix(matrix, len(codomain), len(domain))
        )

    def apply(self, source: Distribution) -> Distribution:
        if source.space != self.domain:
            raise ValueError("source space does not match channel domain")
        return Distribution(self.codomain, self.matrix @ source.probabilities)

    def compose(self, before: Channel) -> Channel:
        """Return ``self ∘ before``."""
        if before.codomain != self.domain:
            raise ValueError("channel codomain/domain mismatch")
        return Channel(before.domain, self.codomain, self.matrix @ before.matrix)

    def tensor(self, other: Channel) -> Channel:
        return Channel(
            self.domain.tensor(other.domain),
            self.codomain.tensor(other.codomain),
            np.kron(self.matrix, other.matrix),
        )

    def marginal(self, axes: int | tuple[int, ...]) -> Channel:
        selected = (axes,) if isinstance(axes, int) else tuple(axes)
        columns = [
            Distribution(self.codomain, self.matrix[:, i]).marginal(selected).probabilities
            for i in range(len(self.domain))
        ]
        codomain = Distribution(self.codomain, self.matrix[:, 0]).marginal(selected).space
        return Channel(self.domain, codomain, np.column_stack(columns))

    def is_deterministic(self, atol: float = ATOL) -> bool:
        zeros_or_ones = np.isclose(self.matrix, 0.0, atol=atol) | np.isclose(
            self.matrix, 1.0, atol=atol
        )
        return bool(np.all(zeros_or_ones) and np.allclose(self.matrix.sum(axis=0), 1.0))

    def __matmul__(self, other: Channel) -> Channel:
        return self.tensor(other)
