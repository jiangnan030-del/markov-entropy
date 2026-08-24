# markov-entropy

[![CI](https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml/badge.svg)](https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml)

An executable implementation of the finite-state constructions in Paolo Perrone's **Markov Categories and Entropy**.

The package models **FinStoch** using column-stochastic matrices and implements:

- finite spaces, probability distributions, and stochastic channels;
- sequential composition, tensor products, copy, discard, joints, and marginals;
- KL, Rényi, and total-variation divergences;
- divergence-induced mutual information and entropy;
- conditional divergence, conditional mutual information, and conditional entropy;
- regression tests for the identities and data-processing inequalities in the paper.

> Scope: v0.1 implements finite state spaces. General measurable spaces (`Stoch`) and differential entropy are deliberately out of scope.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Quick start

```python
from markov_entropy import Distribution, FiniteSpace
from markov_entropy.divergences import KL, Renyi, TotalVariation
from markov_entropy.information import entropy

X = FiniteSpace(["0", "1"])
p = Distribution(X, [0.25, 0.75])

assert entropy(p, KL()) > 0
assert entropy(p, Renyi(alpha=1.0)) == entropy(p, KL())
assert entropy(p, TotalVariation()) == 1 - (0.25**2 + 0.75**2)
```

## Matrix convention

A channel `f: X -> Y` has shape `(len(Y), len(X))`, with
`f.matrix[y, x] = P(y | x)`. Every **column** sums to one, matching the paper.
Composition is ordinary matrix multiplication: `(g >> f)` mathematically means `g ∘ f`, implemented as `g.compose(f)`.

## Mathematical correspondence

| Paper construction | Python API |
|---|---|
| source `p: I -> X` | `Distribution(X, probabilities)` |
| channel `f: X -> Y` | `Channel(X, Y, matrix)` |
| `g ∘ f` | `g.compose(f)` |
| `f ⊗ h` | `f.tensor(h)` |
| copy / discard | `copy(X)` / `discard(X)` |
| joint `fp` | `joint(p, f)` |
| `max_x D(f_x || g_x)` | `channel_divergence(f, g, D)` |
| `D(pXY || pX ⊗ pY)` | `mutual_information(pXY, D)` |
| departure from determinism | `entropy(p_or_f, D)` |

Natural logarithms are used, so information is measured in nats.

## Reproducibility

```bash
python -m pytest
python examples/basic_demo.py
```

## References

- Paolo Perrone, *Markov Categories and Entropy*.
- [DisCoPy](https://github.com/discopy/discopy), for symbolic string-diagram tooling.
- [dit](https://github.com/dit/dit), for broad discrete information-theory functionality.

## License

MIT.
