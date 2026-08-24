import numpy as np

from markov_entropy import Channel, Distribution, FiniteSpace, entropy, mutual_information
from markov_entropy.divergences import KL, TotalVariation, distribution_divergence


def random_distribution(space, rng):
    values = rng.random(len(space))
    return Distribution(space, values / values.sum())


def random_channel(domain, codomain, rng):
    matrix = rng.random((len(codomain), len(domain)))
    matrix /= matrix.sum(axis=0, keepdims=True)
    return Channel(domain, codomain, matrix)


def test_data_processing_and_tensor_additivity_seeded():
    rng = np.random.default_rng(20260824)
    x = FiniteSpace(range(3))
    y = FiniteSpace(range(2))
    p, q = random_distribution(x, rng), random_distribution(x, rng)
    f = random_channel(x, y, rng)
    for divergence in (KL(), TotalVariation()):
        before = distribution_divergence(p, q, divergence)
        after = distribution_divergence(f.apply(p), f.apply(q), divergence)
        assert after <= before + 1e-10
    r, s = random_distribution(y, rng), random_distribution(y, rng)
    lhs = distribution_divergence(p.tensor(r), q.tensor(s), KL())
    rhs = distribution_divergence(p, q, KL()) + distribution_divergence(r, s, KL())
    assert np.isclose(lhs, rhs)


def test_information_and_entropy_data_processing_seeded():
    rng = np.random.default_rng(42)
    x = FiniteSpace(range(2))
    y = FiniteSpace(range(3))
    joint = random_distribution(x.tensor(y), rng)
    fx = random_channel(x, x, rng)
    fy = random_channel(y, y, rng)
    processed = fx.tensor(fy).apply(joint)
    assert mutual_information(processed, KL()) <= mutual_information(joint, KL()) + 1e-10

    deterministic = Channel(y, x, [[1, 0, 1], [0, 1, 0]])
    p = random_distribution(y, rng)
    assert entropy(deterministic.apply(p), KL()) <= entropy(p, KL()) + 1e-10
