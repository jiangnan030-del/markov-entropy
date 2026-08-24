"""Information and randomness as departure from categorical equations."""

from __future__ import annotations

import math

import numpy as np

from .channels import Channel
from .distributions import Distribution
from .divergences.base import Divergence, channel_divergence, distribution_divergence
from .divergences.kl import KL
from .markov import copy, joint


def mutual_information(value: Distribution | Channel, divergence: Divergence) -> float:
    """Measure departure from independence with the chosen divergence."""
    factors = value.space.factors if isinstance(value, Distribution) else value.codomain.factors
    if len(factors) != 2:
        raise ValueError("mutual information requires exactly two output factors")
    if isinstance(value, Distribution):
        independent = value.marginal(0).tensor(value.marginal(1))
        return distribution_divergence(value, independent, divergence)
    first = value.marginal(0)
    second = value.marginal(1)
    columns = [
        np.kron(first.matrix[:, i], second.matrix[:, i])
        for i in range(len(value.domain))
    ]
    independent_channel = Channel(value.domain, value.codomain, np.column_stack(columns))
    return channel_divergence(value, independent_channel, divergence)


def entropy(value: Distribution | Channel, divergence: Divergence) -> float:
    """Measure departure from determinism, following Definition 4.1."""
    if isinstance(value, Distribution):
        diagonal = copy(value.space).apply(value)
        independent = value.tensor(value)
        return distribution_divergence(diagonal, independent, divergence)
    diagonal_channel = copy(value.codomain).compose(value)
    independent_channel = value.tensor(value).compose(copy(value.domain))
    return channel_divergence(diagonal_channel, independent_channel, divergence)


def conditional_divergence(
    first: Channel, second: Channel, source: Distribution, divergence: Divergence
) -> float:
    return distribution_divergence(joint(source, first), joint(source, second), divergence)


def conditional_mutual_information(
    channel: Channel, source: Distribution, divergence: Divergence
) -> float:
    """Divergence from conditional independence, weighted by ``source``."""
    if source.space != channel.domain:
        raise ValueError("source space does not match channel domain")
    factors = channel.codomain.factors
    if len(factors) != 2:
        raise ValueError("conditional mutual information needs two output factors")
    first = channel.marginal(0)
    second = channel.marginal(1)
    independent_columns = [
        np.kron(first.matrix[:, i], second.matrix[:, i])
        for i in range(len(channel.domain))
    ]
    independent = Channel(channel.domain, channel.codomain, np.column_stack(independent_columns))
    return conditional_divergence(channel, independent, source, divergence)


def conditional_entropy(
    channel: Channel, source: Distribution, divergence: Divergence
) -> float:
    """Measure departure from source-almost-sure determinism."""
    if source.space != channel.domain:
        raise ValueError("source space does not match channel domain")
    diagonal = copy(channel.codomain).compose(channel)
    independent = channel.tensor(channel).compose(copy(channel.domain))
    return conditional_divergence(diagonal, independent, source, divergence)


def shannon_entropy(distribution: Distribution) -> float:
    return entropy(distribution, KL())


def renyi_entropy(distribution: Distribution, order: float) -> float:
    if order < 0 or math.isnan(order):
        raise ValueError("order must lie in [0, infinity]")
    p = distribution.probabilities
    if order == 1:
        return shannon_entropy(distribution)
    if order == 0:
        return math.log(float(np.count_nonzero(p)))
    if math.isinf(order):
        return -math.log(float(np.max(p)))
    return math.log(float(np.sum(np.power(p[p > 0], order)))) / (1 - order)


def gini_simpson(distribution: Distribution) -> float:
    return float(1.0 - np.square(distribution.probabilities).sum())
