import math

import numpy as np
import pytest

from markov_entropy.divergences import KL
from markov_entropy.stoch import (
    DensityDistribution,
    density_kl,
    density_renyi,
    density_total_variation,
    estimate_expectation,
    estimate_kl_from_samples,
    partition_lower_bounds,
    validate_normalized,
)


def unit_interval_integrator(points: int = 160):
    nodes, weights = np.polynomial.legendre.leggauss(points)
    locations = (nodes + 1.0) / 2.0
    scaled_weights = weights / 2.0

    def integrate(function):
        return float(
            sum(weight * function(float(location)) for location, weight in zip(locations, scaled_weights))
        )

    return integrate


def test_density_divergences_with_user_supplied_quadrature():
    integrate = unit_interval_integrator()
    uniform = DensityDistribution(lambda _: 0.0, "uniform")
    tilted = DensityDistribution(lambda x: math.log(x + 0.5), "tilted")
    assert validate_normalized(uniform, integrate) == pytest.approx(1.0)
    assert validate_normalized(tilted, integrate) == pytest.approx(1.0)

    expected_kl = -(
        1.5 * math.log(1.5) - 1.5 - (0.5 * math.log(0.5) - 0.5)
    )
    assert density_kl(uniform, tilted, integrate) == pytest.approx(expected_kl)
    assert density_total_variation(uniform, tilted, integrate) == pytest.approx(0.125, abs=1e-4)
    assert density_renyi(uniform, tilted, 0.5, integrate) >= 0.0
    assert density_renyi(uniform, tilted, 1.0, integrate) == pytest.approx(expected_kl)


def test_density_validation_and_explicit_limitations():
    integrate = unit_interval_integrator()
    invalid = DensityDistribution(lambda _: math.log(2.0), "invalid")
    with pytest.raises(ValueError):
        validate_normalized(invalid, integrate)
    with pytest.raises(ValueError):
        validate_normalized(invalid, integrate, tolerance=0.0)
    with pytest.raises(ValueError):
        DensityDistribution(lambda _: math.nan).log_value(0.0)
    with pytest.raises(NotImplementedError):
        density_renyi(invalid, invalid, math.inf, integrate, check_normalized=False)


def test_partition_lower_bounds_and_monotonicity():
    estimate = partition_lower_bounds(
        [
            ([0.5, 0.5], [0.5, 0.5]),
            ([0.4, 0.1, 0.1, 0.4], [0.25, 0.25, 0.25, 0.25]),
        ],
        KL(),
        labels=("coarse", "fine"),
        require_monotone=True,
    )
    assert estimate.values[0] == pytest.approx(0.0)
    assert estimate.lower_bound == estimate.values[1]
    assert estimate.is_monotone

    with pytest.raises(ValueError):
        partition_lower_bounds([], KL())
    with pytest.raises(ValueError):
        partition_lower_bounds([([0.5, 0.5], [1.0])], KL())


def test_sampling_estimates_include_uncertainty():
    samples = np.arange(1.0, 6.0)
    estimate = estimate_expectation(samples, lambda value: value)
    assert estimate.value == pytest.approx(3.0)
    assert estimate.sample_count == 5
    assert estimate.standard_error > 0
    assert estimate.confidence_interval[0] < estimate.value < estimate.confidence_interval[1]

    kl_estimate = estimate_kl_from_samples(
        samples,
        lambda value: -0.5 * value,
        lambda value: -0.75 * value,
    )
    assert kl_estimate.value == pytest.approx(0.75)

    with pytest.raises(ValueError):
        estimate_expectation([1.0], float)
    with pytest.raises(ValueError):
        estimate_expectation([1.0, 2.0], float, confidence=1.0)
