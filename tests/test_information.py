import math

import numpy as np
import pytest

from markov_entropy import (
    Channel,
    Distribution,
    FiniteSpace,
    conditional_entropy,
    conditional_mutual_information,
    entropy,
    gini_simpson,
    mutual_information,
    renyi_entropy,
    shannon_entropy,
)
from markov_entropy.divergences import KL, Renyi, TotalVariation


def test_entropy_recovers_shannon_renyi_and_gini_simpson():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.25, 0.75])
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    assert entropy(p, KL()) == pytest.approx(expected)
    assert shannon_entropy(p) == pytest.approx(expected)
    assert entropy(p, Renyi(0.5)) == pytest.approx(renyi_entropy(p, 1.5))
    assert entropy(p, TotalVariation()) == pytest.approx(gini_simpson(p))


def test_mutual_information_independent_and_correlated():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    assert mutual_information(p.tensor(p), KL()) == pytest.approx(0.0)
    correlated = Distribution(x.tensor(x), [0.5, 0.0, 0.0, 0.5])
    assert mutual_information(correlated, KL()) == pytest.approx(math.log(2))


def test_channel_entropy_and_conditionals():
    x = FiniteSpace([0, 1])
    source = Distribution(x, [0.4, 0.6])
    deterministic = Channel(x, x, np.eye(2))
    noisy = Channel(x, x, [[0.75, 0.25], [0.25, 0.75]])
    assert entropy(deterministic, KL()) == pytest.approx(0.0)
    h0 = shannon_entropy(Distribution(x, noisy.matrix[:, 0]))
    h1 = shannon_entropy(Distribution(x, noisy.matrix[:, 1]))
    expected = 0.4 * h0 + 0.6 * h1
    assert conditional_entropy(noisy, source, KL()) == pytest.approx(expected)


def test_conditional_mutual_information():
    a = FiniteSpace([0, 1])
    x = FiniteSpace([0, 1])
    xy = x.tensor(x)
    source = Distribution(a, [0.5, 0.5])
    independent_columns = np.column_stack([
        np.kron([0.5, 0.5], [0.25, 0.75]),
        np.kron([0.2, 0.8], [0.6, 0.4]),
    ])
    channel = Channel(a, xy, independent_columns)
    assert conditional_mutual_information(channel, source, KL()) == pytest.approx(0.0)
