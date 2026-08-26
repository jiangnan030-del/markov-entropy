import math

import numpy as np
import pytest

from markov_entropy import (
    Channel,
    Distribution,
    FiniteSpace,
    conditional_mutual_information,
    joint,
    mutual_information,
)
from markov_entropy.divergences import KL, Renyi


def binary_entropy(error_probability: float) -> float:
    return -error_probability * math.log(error_probability) - (
        1.0 - error_probability
    ) * math.log(1.0 - error_probability)


def test_binary_symmetric_channel_mutual_information():
    x = FiniteSpace((0, 1))
    source = Distribution(x, [0.5, 0.5])
    error_probability = 0.1
    channel = Channel(
        x,
        x,
        [
            [1.0 - error_probability, error_probability],
            [error_probability, 1.0 - error_probability],
        ],
    )
    expected = math.log(2.0) - binary_entropy(error_probability)
    assert mutual_information(joint(source, channel), KL()) == pytest.approx(expected)


def test_binary_erasure_channel_mutual_information():
    x = FiniteSpace((0, 1))
    y = FiniteSpace((0, 1, "erasure"))
    source = Distribution(x, [0.5, 0.5])
    erasure_probability = 0.25
    channel = Channel(
        x,
        y,
        [
            [1.0 - erasure_probability, 0.0],
            [0.0, 1.0 - erasure_probability],
            [erasure_probability, erasure_probability],
        ],
    )
    expected = (1.0 - erasure_probability) * math.log(2.0)
    assert mutual_information(joint(source, channel), KL()) == pytest.approx(expected)


def test_xor_has_conditional_dependence():
    z = FiniteSpace((0, 1))
    x = FiniteSpace((0, 1))
    source = Distribution(z, [0.5, 0.5])
    xor_conditionals = Channel(
        z,
        x.tensor(x),
        np.column_stack(
            (
                [0.5, 0.0, 0.0, 0.5],
                [0.0, 0.5, 0.5, 0.0],
            )
        ),
    )
    value = conditional_mutual_information(xor_conditionals, source, KL())
    assert value == pytest.approx(math.log(2.0))


def test_renyi_is_stable_for_extreme_probabilities():
    tiny = 1e-300
    p = [1.0 - tiny, tiny]
    q = [tiny, 1.0 - tiny]
    value = Renyi(2.0)(p, q)
    assert math.isfinite(value)
    assert value > 600.0


def test_renyi_near_one_converges_to_kl():
    p = [0.2, 0.3, 0.5]
    q = [0.4, 0.4, 0.2]
    expected = KL()(p, q)
    assert Renyi(1.0 - 1e-9)(p, q) == pytest.approx(expected, rel=1e-8, abs=1e-10)
    assert Renyi(1.0 + 1e-9)(p, q) == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_renyi_disjoint_support_is_infinite_below_one():
    assert math.isinf(Renyi(0.5)([1.0, 0.0], [0.0, 1.0]))
