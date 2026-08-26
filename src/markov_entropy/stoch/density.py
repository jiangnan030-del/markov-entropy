"""Experimental density-based divergences with a user-supplied integrator."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

PointT = TypeVar("PointT")
Integrator = Callable[[Callable[[PointT], float]], float]


@dataclass(frozen=True)
class DensityDistribution(Generic[PointT]):
    """A named log-density with respect to an explicit common base measure.

    The object does not choose a domain, measure, or integration algorithm.
    Those assumptions are supplied by the caller through an ``Integrator``.
    ``-inf`` is accepted as a zero-density value; NaN and ``+inf`` are rejected.
    """

    log_density: Callable[[PointT], float]
    name: str = "density"

    def log_value(self, point: PointT) -> float:
        value = float(self.log_density(point))
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"{self.name} returned an invalid log-density")
        return value

    def value(self, point: PointT) -> float:
        log_value = self.log_value(point)
        if log_value == -math.inf:
            return 0.0
        try:
            return math.exp(log_value)
        except OverflowError:
            return math.inf


def validate_normalized(
    distribution: DensityDistribution[PointT],
    integrator: Integrator[PointT],
    *,
    tolerance: float = 1e-8,
) -> float:
    """Integrate a density and require total mass one within tolerance."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    mass = float(integrator(distribution.value))
    if not math.isfinite(mass) or abs(mass - 1.0) > tolerance:
        raise ValueError(f"{distribution.name} integrates to {mass}, not one")
    return mass


def density_kl(
    first: DensityDistribution[PointT],
    second: DensityDistribution[PointT],
    integrator: Integrator[PointT],
    *,
    check_normalized: bool = True,
    tolerance: float = 1e-8,
) -> float:
    """Compute density KL using a shared base measure and integrator."""
    _check_inputs(first, second, integrator, check_normalized, tolerance)

    def integrand(point: PointT) -> float:
        log_p = first.log_value(point)
        if log_p == -math.inf:
            return 0.0
        log_q = second.log_value(point)
        if log_q == -math.inf:
            return math.inf
        return _exp(log_p) * (log_p - log_q)

    return _nonnegative(float(integrator(integrand)), tolerance)


def density_renyi(
    first: DensityDistribution[PointT],
    second: DensityDistribution[PointT],
    alpha: float,
    integrator: Integrator[PointT],
    *,
    check_normalized: bool = True,
    tolerance: float = 1e-8,
) -> float:
    """Compute finite-order Rényi divergence from densities.

    Order infinity is deliberately unsupported because a generic integrator
    cannot provide an essential supremum. Callers needing that order must
    supply a domain-specific essential-supremum implementation.
    """
    alpha = float(alpha)
    if math.isnan(alpha) or alpha < 0:
        raise ValueError("alpha must lie in [0, infinity]")
    if math.isinf(alpha):
        raise NotImplementedError("alpha=infinity requires an essential-supremum backend")
    if alpha == 1.0:
        return density_kl(
            first,
            second,
            integrator,
            check_normalized=check_normalized,
            tolerance=tolerance,
        )
    _check_inputs(first, second, integrator, check_normalized, tolerance)

    if alpha == 0.0:
        def support_mass(point: PointT) -> float:
            return second.value(point) if first.log_value(point) > -math.inf else 0.0

        mass = float(integrator(support_mass))
        return -math.log(mass) if mass > 0 else math.inf

    def integrand(point: PointT) -> float:
        log_p = first.log_value(point)
        log_q = second.log_value(point)
        if log_p == -math.inf:
            return 0.0
        if log_q == -math.inf:
            return math.inf if alpha > 1 else 0.0
        return _exp(alpha * log_p + (1.0 - alpha) * log_q)

    integral = float(integrator(integrand))
    if integral <= 0:
        return math.inf
    value = math.log(integral) / (alpha - 1.0)
    return _nonnegative(value, tolerance)


def density_total_variation(
    first: DensityDistribution[PointT],
    second: DensityDistribution[PointT],
    integrator: Integrator[PointT],
    *,
    check_normalized: bool = True,
    tolerance: float = 1e-8,
) -> float:
    """Compute one half of the integrated absolute density difference."""
    _check_inputs(first, second, integrator, check_normalized, tolerance)
    value = 0.5 * float(integrator(lambda point: abs(first.value(point) - second.value(point))))
    return _nonnegative(value, tolerance)


def _check_inputs(
    first: DensityDistribution[PointT],
    second: DensityDistribution[PointT],
    integrator: Integrator[PointT],
    check_normalized: bool,
    tolerance: float,
) -> None:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if check_normalized:
        validate_normalized(first, integrator, tolerance=tolerance)
        validate_normalized(second, integrator, tolerance=tolerance)


def _exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _nonnegative(value: float, tolerance: float) -> float:
    if math.isnan(value):
        raise ValueError("integrator returned NaN")
    if value < -tolerance:
        raise ValueError(f"integrator returned a negative divergence: {value}")
    return max(0.0, value)
