"""Finite measurable spaces used as objects of FinStoch."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Hashable, Iterable


@dataclass(frozen=True)
class FiniteSpace:
    """A finite alphabet with unique, ordered labels.

    Tensor products retain their component spaces, enabling unambiguous
    reshaping and marginalisation.
    """

    labels: tuple[Hashable, ...]
    components: tuple["FiniteSpace", ...] = field(default=(), repr=False)

    def __init__(
        self,
        labels: Iterable[Hashable],
        components: tuple["FiniteSpace", ...] = (),
    ) -> None:
        values = tuple(labels)
        if not values:
            raise ValueError("a finite space must contain at least one state")
        if len(set(values)) != len(values):
            raise ValueError("state labels must be unique")
        object.__setattr__(self, "labels", values)
        object.__setattr__(self, "components", components)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def factors(self) -> tuple["FiniteSpace", ...]:
        return self.components or (self,)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(component) for component in self.factors)

    def tensor(self, other: "FiniteSpace") -> "FiniteSpace":
        components = self.factors + other.factors
        labels = tuple(product(*(component.labels for component in components)))
        return FiniteSpace(labels, components=components)

    def __matmul__(self, other: "FiniteSpace") -> "FiniteSpace":
        return self.tensor(other)


UNIT = FiniteSpace(("*",))
