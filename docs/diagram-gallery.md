# DisCoPy diagram gallery

This gallery visualizes the categorical processes implemented numerically by `markov_entropy`. The SVG files are generated from the same finite spaces, distributions, and channels used by the package.

Regenerate the gallery with:

```bash
uv sync --all-extras
uv run python examples/render_discopy_gallery.py
```

## Canonical Markov structure

| Construction | Diagram |
|---|---|
| Channel `f: X → Y` | ![Channel](diagrams/channel.svg) |
| Copy `X → X ⊗ X` | ![Copy](diagrams/copy.svg) |
| Discard `X → I` | ![Discard](diagrams/discard.svg) |
| Composition `g ∘ f` | ![Composition](diagrams/composition.svg) |
| Tensor `f ⊗ g` | ![Tensor](diagrams/tensor.svg) |

## Entropy as departure from determinism

The state entropy construction compares a copied sample with two independent samples.

| Actual process | Reference process |
|---|---|
| ![Copied state](diagrams/state-entropy-actual.svg) | ![Independent states](diagrams/state-entropy-reference.svg) |

For a channel, it compares copying one output with copying the input and applying the channel twice.

| Actual process | Reference process |
|---|---|
| ![Copied channel output](diagrams/channel-entropy-actual.svg) | ![Repeated channel](diagrams/channel-entropy-reference.svg) |

## Mutual information as departure from independence

A joint state is compared with the tensor product of its marginals.

| Actual process | Reference process |
|---|---|
| ![Joint state](diagrams/mutual-information-actual.svg) | ![Product marginals](diagrams/mutual-information-reference.svg) |

A two-output channel is compared with conditionally independent marginal channels sharing the copied input.

| Actual process | Reference process |
|---|---|
| ![Joint channel](diagrams/channel-mutual-information-actual.svg) | ![Conditional product](diagrams/channel-mutual-information-reference.svg) |

The diagrams are symbolic. Numerical matrices, marginals, divergences, and information values continue to be computed by `markov_entropy`.
