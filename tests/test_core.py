import numpy as np
import pytest

from markov_entropy import (
    Channel,
    Distribution,
    FiniteSpace,
    almost_sure_equal,
    copy,
    discard,
    identity,
    is_independent,
    joint,
)


def test_validation_and_immutability():
    x = FiniteSpace(["a", "b"])
    with pytest.raises(ValueError):
        Distribution(x, [0.2, 0.2])
    with pytest.raises(ValueError):
        Channel(x, x, [[0.5, 0.2], [0.6, 0.8]])
    p = Distribution(x, [0.25, 0.75])
    with pytest.raises(ValueError):
        p.probabilities[0] = 0.5


def test_identity_composition_tensor_copy_discard():
    x = FiniteSpace([0, 1])
    f = Channel(x, x, [[0.9, 0.2], [0.1, 0.8]])
    assert np.allclose(identity(x).compose(f).matrix, f.matrix)
    assert np.allclose(f.compose(identity(x)).matrix, f.matrix)
    assert f.tensor(identity(x)).matrix.shape == (4, 4)
    p = Distribution(x, [0.3, 0.7])
    assert np.allclose(discard(x).apply(p).probabilities, [1.0])
    copied = copy(x).apply(p)
    assert np.allclose(copied.probabilities, [0.3, 0.0, 0.0, 0.7])


def test_joint_and_marginals():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    p = Distribution(x, [0.25, 0.75])
    f = Channel(x, y, [[1.0, 0.2], [0.0, 0.8]])
    xy = joint(p, f)
    assert np.allclose(xy.probabilities, [0.25, 0.0, 0.15, 0.60])
    assert np.allclose(xy.marginal(0).probabilities, p.probabilities)
    assert np.allclose(xy.marginal(1).probabilities, f.apply(p).probabilities)


def test_independence_determinism_and_almost_sure_equality():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [1.0, 0.0])
    independent = p.tensor(Distribution(x, [0.4, 0.6]))
    assert is_independent(independent)
    deterministic = identity(x)
    noisy = Channel(x, x, [[1.0, 0.2], [0.0, 0.8]])
    assert deterministic.is_deterministic()
    assert not noisy.is_deterministic()
    assert almost_sure_equal(deterministic, noisy, p)
