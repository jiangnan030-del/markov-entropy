<p align="center">
  <img src="docs/cover.jpg" alt="Markov Entropy — Finite Markov Categories in Python" width="100%">
</p>

<h1 align="center">markov-entropy</h1>

<p align="center">
  <strong>Finite Markov categories, divergences, mutual information, and entropy in Python.</strong>
</p>

<p align="center">
  <a href="https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml"><img src="https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/status-v0.2%20development-blue" alt="v0.2 development">
</p>

<p align="center">
  An executable implementation of the finite-state constructions in Paolo Perrone's
  <em>Markov Categories and Entropy</em>.
</p>

---

## Overview

`markov-entropy` models **FinStoch** with finite spaces and column-stochastic matrices. It connects categorical probability with executable numerical objects, making the paper's constructions available for experiments, regression tests, and reproducible research.

> **Scope.** Version `0.1.0` provides the stable finite-state numerical core. Version `0.2` adds an optional symbolic DisCoPy layer. General measurable spaces (`Stoch`) and differential entropy remain out of scope.

## Highlights

- **Typed probability objects** — finite spaces, probability distributions, and stochastic channels.
- **Markov-category operations** — identity, composition, tensor product, copy, discard, joints, and marginals.
- **Divergences** — KL, Rényi (including `α = 0, 1, ∞`), and total variation.
- **Information quantities** — mutual information, conditional mutual information, entropy, and conditional entropy induced by divergences.
- **Symbolic diagrams** — optional DisCoPy boxes and string diagrams for categorical constructions.
- **Reproducible examples** — five notebooks, a runnable demo, formula-to-code notes, tests, and multi-version CI.

## Installation

Requires Python 3.11 or newer.

```bash
pip install markov-entropy
```

Install the optional DisCoPy adapter:

```bash
pip install "markov-entropy[discopy]"
```

Or from source:

```bash
git clone https://github.com/jiangnan030-del/markov-entropy.git
cd markov-entropy
pip install -e ".[discopy]"
```

For development with uv:

```bash
uv sync --all-extras
uv run pytest
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

## DisCoPy diagrams

```python
from markov_entropy import Channel, FiniteSpace
from markov_entropy.adapters import DiscopyAdapter

X = FiniteSpace([0, 1])
Y = FiniteSpace(["a", "b"])
f = Channel(X, Y, [[0.8, 0.1], [0.2, 0.9]])

adapter = DiscopyAdapter({X: "X", Y: "Y"})
diagram = adapter.channel_box(f, "f")
```

See [`docs/discopy-adapter.md`](docs/discopy-adapter.md) and [`05_discopy_diagrams.ipynb`](examples/05_discopy_diagrams.ipynb).

## Matrix convention

A channel `f: X → Y` has shape `(len(Y), len(X))`, with `f.matrix[y, x] = P(y | x)`. Every column sums to one, matching the paper. `g.compose(f)` represents `g ∘ f`.

## Documentation

- [API reference](docs/api.md)
- [Numerical conventions](docs/conventions.md)
- [Formula-to-code map](docs/formula-map.md)
- [DisCoPy adapter](docs/discopy-adapter.md)

## Examples

| Example | Focus |
|---|---|
| [`01_finstoch_basics.ipynb`](examples/01_finstoch_basics.ipynb) | Finite spaces, distributions, and channels |
| [`02_divergences.ipynb`](examples/02_divergences.ipynb) | KL, Rényi, and total-variation divergences |
| [`03_mutual_information.ipynb`](examples/03_mutual_information.ipynb) | Joint distributions and mutual information |
| [`04_entropy_from_determinism.ipynb`](examples/04_entropy_from_determinism.ipynb) | Entropy as departure from determinism |
| [`05_discopy_diagrams.ipynb`](examples/05_discopy_diagrams.ipynb) | Symbolic Markov string diagrams |

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run mypy src/markov_entropy
uv run pytest
uv run pytest --nbval examples/ -p no:cacheprovider --override-ini="addopts="
uv build
```

GitHub Actions tests Python 3.11, 3.12, and 3.13.

## Roadmap

- [x] FinStoch core objects and Markov-category operations
- [x] KL, Rényi, and total-variation divergences
- [x] Divergence-induced mutual information and entropy
- [x] Tests, notebooks, formula map, and CI
- [x] Optional DisCoPy adapter and symbolic string-diagram layer
- [ ] Rendered diagram gallery and richer visual notebook narrative
- [ ] Explicit experimental backends for selected `Stoch` computations

## Citation

If this software supports your research, cite the repository using [`CITATION.cff`](CITATION.cff) and cite Paolo Perrone's original paper.

## License

Released under the [MIT License](LICENSE).
