<p align="center">
  <img src="docs/cover.png" alt="Markov Entropy — Finite Markov Categories in Python" width="100%">
</p>

<h1 align="center">markov-entropy</h1>

<p align="center">
  <strong>Finite Markov categories, divergences, mutual information, and entropy in Python.</strong>
</p>

<p align="center">
  <a href="https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml"><img src="https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha status">
</p>

<p align="center">
  An executable implementation of the finite-state constructions in Paolo Perrone's
  <em>Markov Categories and Entropy</em>.
</p>

---

## Overview

`markov-entropy` models **FinStoch** with finite spaces and column-stochastic matrices. It connects categorical probability with executable numerical objects, making the paper's constructions available for experiments, regression tests, and reproducible research.

> **Scope.** Version `0.1.0` focuses on finite state spaces. General measurable spaces (`Stoch`) and differential entropy are deliberately out of scope rather than approximated implicitly.

## Highlights

- **Typed probability objects** — finite spaces, probability distributions, and stochastic channels.
- **Markov-category operations** — identity, composition, tensor product, copy, discard, joints, and marginals.
- **Divergences** — KL, Rényi (including `α = 0, 1, ∞`), and total variation.
- **Information quantities** — mutual information, conditional mutual information, entropy, and conditional entropy induced by divergences.
- **Structural checks** — determinism, independence, almost-sure equality, and data-processing regressions.
- **Reproducible examples** — four notebooks, a runnable demo, formula-to-code notes, tests, and multi-version CI.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/jiangnan030-del/markov-entropy.git
cd markov-entropy
pip install -e .
```

For development and testing:

```bash
pip install -e '.[dev]'
pytest
```

For the example notebooks:

```bash
pip install -e '.[notebooks]'
```

## Quick start

```python
from markov_entropy import Distribution, FiniteSpace
from markov_entropy.divergences import KL, Renyi, TotalVariation
from markov_entropy.information import entropy

X = FiniteSpace(["0", "1"])
p = Distribution(X, [0.25, 0.75])

shannon = entropy(p, KL())
renyi = entropy(p, Renyi(alpha=1.0))
gini_simpson = entropy(p, TotalVariation())

assert shannon > 0
assert renyi == shannon
assert gini_simpson == 1 - (0.25**2 + 0.75**2)
```

## Matrix convention

A channel `f: X → Y` has shape `(len(Y), len(X))`, with

```text
f.matrix[y, x] = P(y | x)
```

Every **column** sums to one, matching the convention used in the paper. Composition is ordinary matrix multiplication: `g.compose(f)` represents `g ∘ f`.

## Mathematical correspondence

| Paper construction | Python API |
|---|---|
| Source `p: I → X` | `Distribution(X, probabilities)` |
| Channel `f: X → Y` | `Channel(X, Y, matrix)` |
| Composition `g ∘ f` | `g.compose(f)` |
| Tensor product `f ⊗ h` | `f.tensor(h)` |
| Copy / discard | `copy(X)` / `discard(X)` |
| Joint state `fp` | `joint(p, f)` |
| `maxₓ D(fₓ ‖ gₓ)` | `channel_divergence(f, g, D)` |
| `D(pXY ‖ pX ⊗ pY)` | `mutual_information(pXY, D)` |
| Departure from determinism | `entropy(p_or_f, D)` |

Natural logarithms are used, so information is measured in **nats**. See [`docs/formula-map.md`](docs/formula-map.md) for the formula-to-code index.

## Examples

| Example | Focus |
|---|---|
| [`01_finstoch_basics.ipynb`](examples/01_finstoch_basics.ipynb) | Finite spaces, distributions, and channels |
| [`02_divergences.ipynb`](examples/02_divergences.ipynb) | KL, Rényi, and total-variation divergences |
| [`03_mutual_information.ipynb`](examples/03_mutual_information.ipynb) | Joint distributions and mutual information |
| [`04_entropy_from_determinism.ipynb`](examples/04_entropy_from_determinism.ipynb) | Entropy as departure from determinism |

Run the lightweight script directly:

```bash
python examples/basic_demo.py
```

## Development

```bash
ruff check .
mypy src/markov_entropy
pytest
python examples/basic_demo.py
```

GitHub Actions runs these checks on Python 3.11, 3.12, and 3.13.

## Roadmap

- [x] FinStoch core objects and Markov-category operations
- [x] KL, Rényi, and total-variation divergences
- [x] Divergence-induced mutual information and entropy
- [x] Tests, notebooks, formula map, and CI
- [ ] Optional DisCoPy adapter and string-diagram visualization
- [ ] Explicit experimental backends for selected `Stoch` computations

## Citation

If this software supports your research, cite the repository using [`CITATION.cff`](CITATION.cff) and cite Paolo Perrone's original paper.

## License

Released under the [MIT License](LICENSE).
