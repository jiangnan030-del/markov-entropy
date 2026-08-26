# Numerical and mathematical conventions

## Matrix orientation

A channel `f: X → Y` has matrix shape `(len(Y), len(X))`. Each column is a conditional output distribution and sums to one:

```text
f.matrix[y, x] = P(y | x)
```

Composition is ordinary matrix multiplication. `g.compose(f)` represents `g ∘ f`.

## Logarithm base

All divergences and information quantities use the natural logarithm. Results are measured in nats.

## Validation tolerance

Probability vectors and channel columns use an absolute normalization tolerance of `1e-12` and zero relative tolerance. Tiny negative entries within that tolerance are clipped to zero before normalization. NaN and infinite entries are rejected.

## Support and infinite values

KL divergence follows the conventions:

- `0 log(0/q) = 0`;
- `p > 0` and `q = 0` implies `D(p || q) = +∞`.

Rényi divergence has explicit branches for orders `0`, `1`, and `∞`. Finite non-special orders are evaluated in the log domain. Disjoint support gives positive infinity where required by the standard finite-distribution definition.

## Almost-sure equality

`almost_sure_equal(f, g, p)` compares channel columns whose source mass exceeds the package tolerance. This is a numerical support convention: probabilities at or below `1e-12` are treated as absent.

## Finite scope

Version `0.1.0` implements `FinStoch` only. It does not identify categorical entropy on general measurable spaces with differential entropy. Continuous densities, Radon–Nikodym derivatives, partition suprema, and sampling estimators remain future experimental backends.
