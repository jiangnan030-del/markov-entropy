"""Experimental backends for selected computations beyond finite spaces.

These APIs expose explicit assumptions and approximations. They do not claim
to implement the full category ``Stoch`` or differential entropy.
"""

from .density import (
    DensityDistribution,
    density_kl,
    density_renyi,
    density_total_variation,
    validate_normalized,
)
from .partitions import PartitionDivergenceEstimate, partition_lower_bounds
from .sampling import MonteCarloEstimate, estimate_expectation, estimate_kl_from_samples

__all__ = [
    "DensityDistribution",
    "MonteCarloEstimate",
    "PartitionDivergenceEstimate",
    "density_kl",
    "density_renyi",
    "density_total_variation",
    "estimate_expectation",
    "estimate_kl_from_samples",
    "partition_lower_bounds",
    "validate_normalized",
]
