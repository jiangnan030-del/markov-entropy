"""Canonical FinStoch operations."""

from __future__ import annotations

import numpy as np

from .channels import Channel
from .distributions import Distribution
from .spaces import FiniteSpace, UNIT
from .validation import ATOL


def identity(space: FiniteSpace) -> Channel:
    return Channel(space, space, np.eye(len(space)))


def copy(space: FiniteSpace) -> Channel:
    codomain = space.tensor(space)
    matrix = np.zeros((len(codomain), len(space)))
    for index in range(len(space)):
        matrix[index * len(space) + index, index] = 1.0
    return Channel(space, codomain, matrix)


def discard(space: FiniteSpace) -> Channel:
    return Channel(space, UNIT, np.ones((1, len(space))))


def joint(source: Distribution, channel: Channel) -> Distribution:
    """Return ``P(x, y) = P(x) P(y | x)``."""
    if source.space != channel.domain:
        raise ValueError("source space does not match channel domain")
    values = (channel.matrix * source.probabilities[np.newaxis, :]).T.reshape(-1)
    return Distribution(source.space.tensor(channel.codomain), values)


def is_independent(distribution: Distribution, atol: float = ATOL) -> bool:
    factors = distribution.space.factors
    if len(factors) < 2:
        raise ValueError("independence requires a product space")
    product = Distribution.product(
        *(distribution.marginal(axis) for axis in range(len(factors)))
    )
    return bool(np.allclose(distribution.probabilities, product.probabilities, atol=atol))


def almost_sure_equal(
    first: Channel, second: Channel, source: Distribution, atol: float = ATOL
) -> bool:
    if first.domain != second.domain or first.codomain != second.codomain:
        raise ValueError("channels must have the same type")
    if source.space != first.domain:
        raise ValueError("source space does not match channel domain")
    support = source.probabilities > atol
    return bool(np.allclose(first.matrix[:, support], second.matrix[:, support], atol=atol))
