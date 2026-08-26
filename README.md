<p align="center">
  <img src="docs/cover.jpg" alt="Markov Entropy — Finite Markov Categories in Python" width="100%">
</p>

<h1 align="center">markov-entropy</h1>

<p align="center">
  <strong>Finite Markov categories and experimental general-space information calculations in Python.</strong>
</p>

<p align="center">
  <a href="https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml"><img src="https://github.com/jiangnan030-del/markov-entropy/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/version-0.3.0-blue" alt="Version 0.3.0">
</p>

<p align="center">
  An executable implementation of finite-state constructions from Paolo Perrone's
  <em>Markov Categories and Entropy</em>.
</p>

---

## Overview

`markov-entropy` models **FinStoch** with finite spaces and column-stochastic matrices. Version 0.2 added an optional DisCoPy layer for symbolic Markov string diagrams. Version 0.3 adds explicitly experimental density, finite-partition, and sampling calculations while keeping the stable finite API unchanged.

> **Scope.** Finite state spaces are the stable core. Experimental calculations live under `markov_entropy.stoch`; they do not constitute a full implementation of general measurable spaces, arbitrary kernels, Radon–Nikodym derivatives, exact partition suprema, or differential entropy.

## Highlights

- Immutable finite spaces, probability distributions, and stochastic channels.
- Identity, composition, tensor product, copy, discard, joints, and marginals.
- KL, Rényi, and total-variation divergences.
- Mutual information, conditional mutual information, entropy, and conditional entropy.
- Optional DisCoPy boxes and string diagrams for categorical constructions.
- Experimental density divergences using a caller-supplied common-base-measure integrator.
- Finite-partition lower-bound sequences with monotonicity diagnostics.
- Monte Carlo expectation and KL estimates with standard errors and confidence intervals.
- Reproducible SVG gallery, six notebooks, theorem tests, and Python 3.11–3.13 CI.

## Installation

```bash
pip install markov-entropy
```

Install string-diagram support:

```bash
pip install "markov-entropy[discopy]"
```

For development:

```bash
git clone https://github.com/jiangnan030-del/markov-entropy.git
cd markov-entropy
uv sync --all-extras
uv run pytest
```

## Quick start

```python
from markov_entropy import Distribution, FiniteSpace
from markov_entropy.divergences import KL
from markov_entropy.information import entropy

X = FiniteSpace(["0", "1"])
p = Distribution(X, [0.25, 0.75])
assert entropy(p, KL()) > 0
```

## String diagrams

```python
from markov_entropy import Channel, FiniteSpace
from markov_entropy.adapters import DiscopyAdapter

X = FiniteSpace([0, 1])
Y = FiniteSpace(["a", "b"])
f = Channel(X, Y, [[0.8, 0.1], [0.2, 0.9]])
adapter = DiscopyAdapter({X: "X", Y: "Y"})
diagram = adapter.channel_box(f, "f")
```

Render the complete SVG gallery:

```bash
uv run python examples/render_discopy_gallery.py
```

## Experimental Stoch calculations

```python
import math

from markov_entropy.stoch import DensityDistribution, density_kl

p = DensityDistribution(lambda _: 0.0, "uniform")
q = DensityDistribution(lambda x: math.log(x + 0.5), "tilted")

# The caller explicitly supplies the domain and quadrature rule.
value = density_kl(p, q, integrator)
```

The experimental namespace also provides finite-partition lower-bound sequences and Monte Carlo estimates with reported uncertainty. See the [experimental Stoch guide](docs/stoch-experimental.md) and [notebook](examples/06_stoch_experimental.ipynb).

## Documentation

- [API reference](docs/api.md)
- [Numerical conventions](docs/conventions.md)
- [Formula-to-code map](docs/formula-map.md)
- [DisCoPy adapter](docs/discopy-adapter.md)
- [Diagram gallery](docs/diagram-gallery.md)
- [Experimental Stoch backends](docs/stoch-experimental.md)
- [Changelog](CHANGELOG.md)

## Validation

```bash
uv run ruff check .
uv run mypy src/markov_entropy
uv run pytest
uv run pytest --nbval examples/ -p no:cacheprovider --override-ini="addopts="
uv run python examples/render_discopy_gallery.py --output build/diagrams
uv build
```

## Roadmap

- [x] FinStoch numerical core
- [x] Divergence-induced information and entropy
- [x] Paper-theorem and property-based validation
- [x] Optional DisCoPy adapter
- [x] Reproducible SVG diagram gallery and visual notebook
- [x] Initial experimental density, partition, and sampling interfaces
- [ ] Domain-specific integrators, essential suprema, dependent-sample errors, and measurable kernels

## Citation

If this software supports your research, cite the repository using [`CITATION.cff`](CITATION.cff) and cite Paolo Perrone's original paper.

## License

Released under the [MIT License](LICENSE).
