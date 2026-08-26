"""Sampling estimators with explicit uncertainty for experimental Stoch work."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Generic, TypeVar

import numpy as np

PointT = TypeVar("PointT")


@dataclass(frozen=True)
class MonteCarloEstimate(Generic[PointT]):
    """A scalar Monte Carlo estimate and normal-approximation interval."""

    value: float
    standard_error: float
    confidence_interval: tuple[float, float]
    sample_count: int
    confidence: float


def estimate_expectation(
    samples: Iterable[PointT],
    statistic: Callable[[PointT], float],
    *,
    confidence: float = 0.95,
) -> MonteCarloEstimate[PointT]:
    """Estimate an expectation from independent samples.

    The interval is a normal approximation and is not a finite-sample
    guarantee. Correlated samples require a domain-specific error estimator.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    values = np.asarray([float(statistic(sample)) for sample in samples], dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two scalar samples are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample statistics must be finite")

    value = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    interval = (
        value - critical * standard_error,
        value + critical * standard_error,
    )
    return MonteCarloEstimate(value, standard_error, interval, int(values.size), confidence)


def estimate_kl_from_samples(
    samples: Iterable[PointT],
    first: Callable[[PointT], float],
    second: Callable[[PointT], float],
    *,
    confidence: float = 0.95,
) -> MonteCarloEstimate[PointT]:
    """Estimate ``E_p[log p(X) - log q(X)]`` from samples drawn from ``p``."""
    return estimate_expectation(
        samples,
        lambda sample: float(first(sample)) - float(second(sample)),
        confidence=confidence,
    )
