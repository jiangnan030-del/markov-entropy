# DisCoPy adapter

The optional DisCoPy adapter adds a symbolic string-diagram layer while keeping all numerical probability and information calculations in `markov_entropy`.

## Installation

```bash
pip install "markov-entropy[discopy]"
```

The v0.2 adapter targets DisCoPy `1.2.2`, which supports the project's Python 3.11 baseline.

## Basic usage

```python
from markov_entropy import Channel, Distribution, FiniteSpace
from markov_entropy.adapters import DiscopyAdapter

X = FiniteSpace([0, 1])
Y = FiniteSpace(["a", "b"])
p = Distribution(X, [0.4, 0.6])
f = Channel(X, Y, [[0.8, 0.1], [0.2, 0.9]])

adapter = DiscopyAdapter({X: "X", Y: "Y"})
box = adapter.channel_box(f, "f")
entropy_pair = adapter.entropy_diagrams(p)

entropy_pair.actual.draw(path="entropy-copy.png")
entropy_pair.reference.draw(path="entropy-independent.png")
```

## Representation choices

- An atomic `FiniteSpace` becomes one DisCoPy `markov.Ty` wire.
- A product space becomes the tensor product of its factor wires.
- A numerical `Channel` becomes a symbolic `markov.Box` with matching domain and codomain.
- Matrices are not copied into DisCoPy; `markov_entropy` remains the numerical source of truth.
- Copy and discard use DisCoPy's canonical Markov diagrams.

## Available constructions

- `type_for` — finite space to DisCoPy type.
- `channel_box` and `state_box` — symbolic boxes.
- `copy_diagram` and `discard_diagram` — canonical Markov structure.
- `compose_diagram` and `tensor_diagram` — sequential and parallel composition.
- `entropy_diagrams` — copied state versus independent copies.
- `channel_entropy_diagrams` — copied output versus repeated channel.
- `mutual_information_diagrams` — joint state versus product marginals.
- `channel_mutual_information_diagrams` — joint channel versus conditionally independent outputs.

`DiagramComparison.draw` writes the actual and reference processes to separate files so they can be compared in documentation or notebooks.
