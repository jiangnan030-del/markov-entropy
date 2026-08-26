# API reference

`markov-entropy` implements finite Markov categories with column-stochastic matrices. All logarithmic information quantities use natural logarithms and are measured in nats.

## Core objects

### `FiniteSpace(labels)`

An immutable, ordered finite alphabet. Labels must be non-empty, unique, and hashable. `space.tensor(other)` constructs the Cartesian product and preserves factor metadata for marginalization.

### `Distribution(space, probabilities)`

A probability source on a `FiniteSpace`. The vector must be finite, non-negative within numerical tolerance, and sum to one. Stored arrays are normalized and read-only.

Important methods:

- `Distribution.product(*distributions)` — product distribution.
- `distribution.tensor(other)` — tensor/product distribution.
- `distribution.marginal(axes)` — retain selected product axes.

### `Channel(domain, codomain, matrix)`

A column-stochastic channel. Its matrix shape is `(len(codomain), len(domain))`, and `matrix[y, x] = P(y | x)`.

Important methods:

- `channel.apply(source)` — push a distribution through the channel.
- `channel.compose(before)` — return `channel ∘ before`.
- `channel.tensor(other)` — parallel composition.
- `channel.marginal(axes)` — retain selected codomain factors.
- `channel.is_deterministic()` — test whether every column is a point mass.

## Canonical Markov operations

- `identity(space)` — identity channel.
- `copy(space)` — deterministic diagonal channel `x ↦ (x, x)`.
- `discard(space)` — unique channel to the unit space.
- `joint(source, channel)` — joint state `P(x, y) = P(x)P(y | x)`.
- `is_independent(distribution)` — compare a product-space state with the product of its marginals.
- `almost_sure_equal(first, second, source)` — compare channels on the numerical support of a source.

## Divergences

### `KL()`

Computes `D(p || q) = Σ p log(p/q)`. Terms with `p = 0` contribute zero. If `p > 0` where `q = 0`, the result is positive infinity.

### `Renyi(alpha)`

Computes Rényi divergence for `alpha ∈ [0, ∞]`. Orders `0`, `1`, and `∞` use explicit formulas. Other finite orders are evaluated in the log domain, with a cancellation-resistant formula near `alpha = 1`.

### `TotalVariation()`

Computes `0.5 * Σ |p - q|`.

### Divergence lifting

- `distribution_divergence(first, second, divergence)` — divergence between aligned distributions.
- `channel_divergence(first, second, divergence)` — maximum column divergence over inputs.
- `conditional_divergence(first, second, source, divergence)` — divergence between source-weighted joint states.

A custom divergence only needs to implement `__call__(first, second) -> float`.

## Information quantities

- `mutual_information(value, divergence)` — departure from independence for a two-factor state or channel.
- `conditional_mutual_information(channel, source, divergence)` — source-weighted departure from conditional independence.
- `entropy(value, divergence)` — departure from determinism for a state or channel.
- `conditional_entropy(channel, source, divergence)` — source-weighted departure from determinism.
- `shannon_entropy(distribution)` — KL-induced entropy.
- `renyi_entropy(distribution, order)` — conventional finite Rényi entropy.
- `gini_simpson(distribution)` — `1 - Σ p²`.

## Optional DisCoPy adapter

`markov_entropy.adapters.DiscopyAdapter` maps finite spaces and channels to symbolic Markov string diagrams. The numerical matrices remain in `markov_entropy`. See the [DisCoPy adapter guide](discopy-adapter.md).

## Experimental Stoch namespace

These APIs expose explicit assumptions and approximations. They are deliberately separated from the stable finite categorical API.

### Density calculations

- `DensityDistribution(log_density, name)` — a log-density relative to a caller-defined base measure.
- `validate_normalized(distribution, integrator)` — check total mass under a supplied integrator.
- `density_kl(first, second, integrator)` — density KL.
- `density_renyi(first, second, alpha, integrator)` — finite-order density Rényi divergence.
- `density_total_variation(first, second, integrator)` — integrated total variation.

### Finite partitions

- `partition_lower_bounds(levels, divergence)` — evaluate divergence on user-supplied finite coarse-grainings.
- `PartitionDivergenceEstimate` — values, labels, largest observed lower bound, and monotonicity diagnostic.

### Sampling estimates

- `estimate_expectation(samples, statistic)` — sample mean, standard error, and normal interval.
- `estimate_kl_from_samples(samples, first, second)` — estimate a log-density ratio expectation from samples drawn from the first distribution.
- `MonteCarloEstimate` — point estimate and uncertainty metadata.

See [Experimental Stoch backends](stoch-experimental.md) for assumptions and non-goals.

## Exceptions and numerical behavior

Public constructors raise `ValueError` for incompatible spaces, invalid shapes, non-finite entries, negative probabilities, or failed normalization. See [Numerical conventions](conventions.md) for tolerance, support, infinity handling, and experimental-backend assumptions.
