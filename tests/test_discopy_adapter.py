import numpy as np
import pytest

pytest.importorskip("discopy")

from markov_entropy import Channel, Distribution, FiniteSpace
from markov_entropy.adapters import DiscopyAdapter


def make_objects():
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b"))
    source = Distribution(x, [0.4, 0.6])
    channel = Channel(x, y, [[0.8, 0.1], [0.2, 0.9]])
    return x, y, source, channel


def test_space_registration_and_product_types():
    x, y, _, _ = make_objects()
    adapter = DiscopyAdapter()
    adapter.register_space(x, "X")
    adapter.register_space(y, "Y")
    assert str(adapter.type_for(x)) == "X"
    assert str(adapter.type_for(y)) == "Y"
    assert str(adapter.type_for(x.tensor(y))) == "X @ Y"

    with pytest.raises(ValueError):
        adapter.register_space(x.tensor(y), "XY")
    with pytest.raises(ValueError):
        adapter.register_space(x, "")


def test_channel_boxes_composition_and_tensor():
    x, y, _, channel = make_objects()
    reverse = Channel(y, x, np.eye(2))
    adapter = DiscopyAdapter({x: "X", y: "Y"})

    box = adapter.channel_box(channel, "f")
    assert str(box.dom) == "X"
    assert str(box.cod) == "Y"

    composed = adapter.compose_diagram(channel, reverse, "f", "g")
    assert composed.dom == adapter.type_for(x)
    assert composed.cod == adapter.type_for(x)

    parallel = adapter.tensor_diagram(channel, reverse)
    assert parallel.dom == adapter.type_for(x.tensor(y))
    assert parallel.cod == adapter.type_for(y.tensor(x))

    with pytest.raises(ValueError):
        adapter.compose_diagram(channel, channel)


def test_copy_discard_and_entropy_diagrams():
    x, _, source, channel = make_objects()
    adapter = DiscopyAdapter({x: "X", channel.codomain: "Y"})
    copied = adapter.copy_diagram(x)
    discarded = adapter.discard_diagram(x)
    assert copied.dom == adapter.type_for(x)
    assert copied.cod == adapter.type_for(x.tensor(x))
    assert discarded.dom == adapter.type_for(x)
    assert len(discarded.cod) == 0

    state_comparison = adapter.entropy_diagrams(source)
    assert state_comparison.actual.dom == state_comparison.reference.dom
    assert state_comparison.actual.cod == state_comparison.reference.cod

    channel_comparison = adapter.channel_entropy_diagrams(channel)
    assert channel_comparison.actual.dom == channel_comparison.reference.dom
    assert channel_comparison.actual.cod == channel_comparison.reference.cod


def test_mutual_information_diagrams():
    x, y, source, channel = make_objects()
    adapter = DiscopyAdapter({x: "X", y: "Y"})
    product = source.tensor(Distribution(y, [0.25, 0.75]))
    state_comparison = adapter.mutual_information_diagrams(product)
    assert state_comparison.actual.dom == state_comparison.reference.dom
    assert state_comparison.actual.cod == state_comparison.reference.cod

    joint_channel = Channel(
        x,
        y.tensor(y),
        [[0.6, 0.1], [0.2, 0.2], [0.1, 0.3], [0.1, 0.4]],
    )
    channel_comparison = adapter.channel_mutual_information_diagrams(joint_channel)
    assert channel_comparison.actual.dom == channel_comparison.reference.dom
    assert channel_comparison.actual.cod == channel_comparison.reference.cod

    with pytest.raises(ValueError):
        adapter.mutual_information_diagrams(source)
    with pytest.raises(ValueError):
        adapter.channel_mutual_information_diagrams(channel)
