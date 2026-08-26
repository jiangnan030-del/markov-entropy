import numpy as np
from hypothesis import given, settings, strategies as st

from markov_entropy import Channel, Distribution, FiniteSpace, entropy, mutual_information
from markov_entropy.divergences import (
    KL,
    Renyi,
    TotalVariation,
    channel_divergence,
    distribution_divergence,
)


@st.composite
def probability_values(draw, size=3):
    weights = draw(
        st.lists(st.integers(min_value=1, max_value=1000), min_size=size, max_size=size)
    )
    values = np.asarray(weights, dtype=np.float64)
    return values / values.sum()


@st.composite
def stochastic_matrices(draw, rows=3, columns=3):
    columns_data = draw(
        st.lists(
            st.lists(
                st.integers(min_value=1, max_value=1000),
                min_size=rows,
                max_size=rows,
            ),
            min_size=columns,
            max_size=columns,
        )
    )
    matrix = np.asarray(columns_data, dtype=np.float64).T
    return matrix / matrix.sum(axis=0, keepdims=True)


DIVERGENCES = (KL(), Renyi(0.5), Renyi(2.0), TotalVariation())


@settings(max_examples=40, deadline=None)
@given(probability_values(), probability_values())
def test_divergences_are_nonnegative_and_reflexive(p_values, q_values):
    x = FiniteSpace(range(3))
    p = Distribution(x, p_values)
    q = Distribution(x, q_values)
    for divergence in DIVERGENCES:
        assert distribution_divergence(p, q, divergence) >= -1e-12
        assert abs(distribution_divergence(p, p, divergence)) <= 1e-12


@settings(max_examples=40, deadline=None)
@given(probability_values(), probability_values(), stochastic_matrices(rows=2, columns=3))
def test_distribution_data_processing(p_values, q_values, matrix):
    x = FiniteSpace(range(3))
    y = FiniteSpace(range(2))
    p = Distribution(x, p_values)
    q = Distribution(x, q_values)
    channel = Channel(x, y, matrix)
    for divergence in DIVERGENCES:
        before = distribution_divergence(p, q, divergence)
        after = distribution_divergence(channel.apply(p), channel.apply(q), divergence)
        assert after <= before + 1e-10


@settings(max_examples=30, deadline=None)
@given(
    probability_values(),
    probability_values(),
    probability_values(size=2),
    probability_values(size=2),
)
def test_kl_tensor_additivity(p_values, q_values, r_values, s_values):
    x = FiniteSpace(range(3))
    y = FiniteSpace(range(2))
    p, q = Distribution(x, p_values), Distribution(x, q_values)
    r, s = Distribution(y, r_values), Distribution(y, s_values)
    lhs = distribution_divergence(p.tensor(r), q.tensor(s), KL())
    rhs = distribution_divergence(p, q, KL()) + distribution_divergence(r, s, KL())
    assert np.isclose(lhs, rhs, atol=1e-10, rtol=1e-10)


@settings(max_examples=30, deadline=None)
@given(
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=2, columns=3),
    stochastic_matrices(rows=2, columns=2),
)
def test_composition_identity_and_associativity(f_matrix, g_matrix, h_matrix):
    x = FiniteSpace(range(2))
    y = FiniteSpace(range(3))
    z = FiniteSpace(range(2))
    w = FiniteSpace(("left", "right"))
    f = Channel(x, y, f_matrix)
    g = Channel(y, z, g_matrix)
    h = Channel(z, w, h_matrix)
    left = h.compose(g).compose(f)
    right = h.compose(g.compose(f))
    assert np.allclose(left.matrix, right.matrix)


@settings(max_examples=30, deadline=None)
@given(
    stochastic_matrices(rows=2, columns=2),
    stochastic_matrices(rows=2, columns=2),
    stochastic_matrices(rows=2, columns=2),
    stochastic_matrices(rows=2, columns=2),
)
def test_tensor_interchange_law(f_matrix, g_matrix, u_matrix, v_matrix):
    x = FiniteSpace(("x0", "x1"))
    y = FiniteSpace(("y0", "y1"))
    z = FiniteSpace(("z0", "z1"))
    a = FiniteSpace(("a0", "a1"))
    b = FiniteSpace(("b0", "b1"))
    c = FiniteSpace(("c0", "c1"))
    f, g = Channel(x, y, f_matrix), Channel(y, z, g_matrix)
    u, v = Channel(a, b, u_matrix), Channel(b, c, v_matrix)
    left = g.compose(f).tensor(v.compose(u))
    right = g.tensor(v).compose(f.tensor(u))
    assert np.allclose(left.matrix, right.matrix)


@settings(max_examples=30, deadline=None)
@given(
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=2, columns=3),
)
def test_channel_divergence_contracts_under_postprocessing(f_matrix, g_matrix, h_matrix):
    x = FiniteSpace(range(2))
    y = FiniteSpace(range(3))
    z = FiniteSpace(("z0", "z1"))
    f, g = Channel(x, y, f_matrix), Channel(x, y, g_matrix)
    h = Channel(y, z, h_matrix)
    for divergence in DIVERGENCES:
        before = channel_divergence(f, g, divergence)
        after = channel_divergence(h.compose(f), h.compose(g), divergence)
        assert after <= before + 1e-10


@settings(max_examples=30, deadline=None)
@given(
    probability_values(size=4),
    probability_values(size=4),
)
def test_marginalization_does_not_increase_divergence(p_values, q_values):
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b"))
    p = Distribution(x.tensor(y), p_values)
    q = Distribution(x.tensor(y), q_values)
    for divergence in DIVERGENCES:
        before = distribution_divergence(p, q, divergence)
        after = distribution_divergence(p.marginal(0), q.marginal(0), divergence)
        assert after <= before + 1e-10


@settings(max_examples=30, deadline=None)
@given(
    probability_values(size=4),
    stochastic_matrices(rows=2, columns=2),
    stochastic_matrices(rows=2, columns=2),
)
def test_mutual_information_data_processing(joint_values, fx_matrix, fy_matrix):
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b"))
    joint = Distribution(x.tensor(y), joint_values)
    fx = Channel(x, x, fx_matrix)
    fy = Channel(y, y, fy_matrix)
    processed = fx.tensor(fy).apply(joint)
    assert mutual_information(processed, KL()) <= mutual_information(joint, KL()) + 1e-10


@settings(max_examples=30, deadline=None)
@given(probability_values())
def test_deterministic_postprocessing_does_not_increase_entropy(p_values):
    x = FiniteSpace(range(3))
    y = FiniteSpace((0, 1))
    source = Distribution(x, p_values)
    deterministic = Channel(x, y, [[1, 0, 1], [0, 1, 0]])
    assert entropy(deterministic.apply(source), KL()) <= entropy(source, KL()) + 1e-10
