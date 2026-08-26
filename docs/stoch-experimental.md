# Experimental Stoch backends

Version 0.3 development introduces explicit approximation interfaces for selected calculations beyond finite state spaces. These APIs do **not** claim to implement the full category `Stoch`, and they do not identify categorical entropy with differential entropy.

## Density backend

`DensityDistribution` stores a user-supplied log-density. The caller must also supply an integrator representing a common base measure and domain.

```python
import math

from markov_entropy.stoch import DensityDistribution, density_kl

p = DensityDistribution(lambda x: 0.0, "uniform")
q = DensityDistribution(lambda x: math.log(x + 0.5), "tilted")

# The caller owns the quadrature rule and integration domain.
value = density_kl(p, q, integrator)
```

Available functions:

- `density_kl`
- `density_renyi` for finite orders
- `density_total_variation`
- `validate_normalized`

Rényi order infinity is intentionally unsupported because a generic integrator cannot compute an essential supremum.

## Finite-partition backend

`partition_lower_bounds` evaluates a divergence on a sequence of finite coarse-grainings. For refining partitions and suitable divergences, these values can form lower bounds approaching the general-space divergence.

The function cannot infer whether one partition refines another. Callers must provide this structural guarantee; `require_monotone=True` only checks the resulting numerical sequence.

## Sampling backend

`estimate_expectation` and `estimate_kl_from_samples` return:

- point estimate;
- standard error;
- normal-approximation confidence interval;
- sample count and confidence level.

These are estimators, not exact categorical quantities. Independent samples are assumed. MCMC, importance sampling, heavy tails, and dependent data require domain-specific uncertainty methods.

## Non-goals

The experimental module does not currently provide:

- arbitrary measurable-space objects or kernels;
- automatic Radon–Nikodym derivatives;
- exact suprema over all countable partitions;
- differential entropy;
- essential-supremum Rényi divergence;
- guaranteed finite-sample confidence bounds.
