import math

import numpy as np
import pytest

from markov_entropy import Channel, Distribution, FiniteSpace
from markov_entropy.divergences import (
    KL,
    Renyi,
    TotalVariation,
    channel_divergence,
    distribution_divergence,
)


def test_kl_edges_and_known_value():
    kl = KL()
    assert kl([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert math.isinf(kl([1.0, 0.0], [0.0, 1.0]))


def test_renyi_special_orders():
    p, q = [0.75, 0.25], [0.5, 0.5]
    assert Renyi(1)(p, q) == pytest.approx(KL()(p, q))
    assert Renyi(0)(p, q) == pytest.approx(0.0)
    assert Renyi(float("inf"))(p, q) == pytest.approx(math.log(1.5))
    with pytest.raises(ValueError):
        Renyi(-1)


def test_total_variation():
    assert TotalVariation()([1, 0], [0, 1]) == pytest.approx(1.0)
    assert TotalVariation()([0.3, 0.7], [0.2, 0.8]) == pytest.approx(0.1)


def test_distribution_and_channel_divergence():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    q = Distribution(x, [0.25, 0.75])
    assert distribution_divergence(p, q, KL()) == pytest.approx(KL()(p.probabilities, q.probabilities))
    f = Channel(x, x, np.eye(2))
    g = Channel(x, x, [[0.9, 0.2], [0.1, 0.8]])
    assert channel_divergence(f, g, TotalVariation()) == pytest.approx(0.2)
