"""Tests targeting edge cases and error branches for coverage."""

import math

import numpy as np
import pytest

from markov_entropy import (
    Channel,
    Distribution,
    FiniteSpace,
    almost_sure_equal,
    conditional_entropy,
    conditional_mutual_information,
    entropy,
    gini_simpson,
    is_independent,
    joint,
    mutual_information,
    renyi_entropy,
    shannon_entropy,
)
from markov_entropy.divergences import (
    KL,
    Renyi,
    channel_divergence,
    distribution_divergence,
)
from markov_entropy.divergences.base import aligned_probabilities
from markov_entropy.validation import probability_vector, stochastic_matrix

# --- spaces.py coverage ---


def test_space_empty_raises():
    with pytest.raises(ValueError, match="at least one state"):
        FiniteSpace([])


def test_space_duplicate_labels_raise():
    with pytest.raises(ValueError, match="unique"):
        FiniteSpace(["a", "a"])


def test_space_matmul():
    x = FiniteSpace([0, 1])
    xy = x @ x
    assert len(xy) == 4


# --- distributions.py coverage ---


def test_distribution_product_empty_raises():
    with pytest.raises(ValueError, match="at least one distribution"):
        Distribution.product()


def test_distribution_marginal_empty_axes_raises():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    with pytest.raises(ValueError, match="at least one marginal axis"):
        p.marginal([])


def test_distribution_marginal_duplicate_axes_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    p = Distribution(x.tensor(y), [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="unique"):
        p.marginal([0, 0])


def test_distribution_marginal_axis_out_of_range_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    p = Distribution(x.tensor(y), [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="out of range"):
        p.marginal(5)


def test_distribution_marginal_multi_axis_reorder():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    z = FiniteSpace(["p", "q"])
    p = Distribution(x.tensor(y).tensor(z), [0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0])
    # Marginalise axes (2, 0) — should reorder to (z, x)
    m = p.marginal([2, 0])
    assert len(m.space) == 4
    assert m.space.factors[0] == z
    assert m.space.factors[1] == x


def test_distribution_matmul():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.3, 0.7])
    q = Distribution(x, [0.5, 0.5])
    pq = p @ q
    assert len(pq.space) == 4


# --- channels.py coverage ---


def test_channel_apply_domain_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    f = Channel(x, y, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="domain"):
        f.apply(Distribution(y, [0.5, 0.5]))


def test_channel_compose_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    z = FiniteSpace(["x", "y"])
    g = Channel(y, z, [[0.5, 0.5], [0.5, 0.5]])
    h = Channel(z, x, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="mismatch"):
        g.compose(h)  # h.codomain=z != g.domain=y


def test_channel_matmul():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    f = Channel(x, y, [[1.0, 0.0], [0.0, 1.0]])
    g = Channel(x, y, [[0.5, 0.5], [0.5, 0.5]])
    fg = f @ g
    assert fg.matrix.shape == (4, 4)


# --- divergences/base.py coverage ---


def test_aligned_probabilities_shape_mismatch():
    with pytest.raises(ValueError, match="one-dimensional and aligned"):
        aligned_probabilities([0.5, 0.5], [0.5, 0.5, 0.0])


def test_aligned_probabilities_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        aligned_probabilities([1.5, -0.5], [0.5, 0.5])


def test_aligned_probabilities_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        aligned_probabilities([0.3, 0.3], [0.5, 0.5])


def test_distribution_divergence_space_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    p = Distribution(x, [0.5, 0.5])
    q = Distribution(y, [0.5, 0.5])
    with pytest.raises(ValueError, match="same space"):
        distribution_divergence(p, q, KL())


def test_channel_divergence_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    f = Channel(x, y, [[1.0, 0.0], [0.0, 1.0]])
    g = Channel(x, x, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="domain and codomain"):
        channel_divergence(f, g, KL())


# --- divergences/renyi.py coverage ---


def test_renyi_inf_with_support_mismatch():
    # alpha = inf, p > 0 and q == 0 → inf
    assert math.isinf(Renyi(float("inf"))([1.0, 0.0], [0.0, 1.0]))


def test_renyi_alpha_gt_1_with_support_mismatch():
    # alpha > 1, p > 0 and q == 0 → inf
    assert math.isinf(Renyi(2.0)([1.0, 0.0], [0.0, 1.0]))


def test_renyi_total_zero_returns_inf():
    # alpha in (0,1), all mass where q>0 but total sums to 0 is impossible
    # with valid probabilities; test the inf branch via p with zeros
    # When p > 0 and q > 0 but total is 0 — not reachable with real probs,
    # but we can test alpha=0 with zero mass
    assert Renyi(0.0)([0.0, 1.0], [1.0, 0.0]) == float("inf")


# --- information.py coverage ---


def test_mutual_information_channel_zero():
    x = FiniteSpace([0, 1])
    # Deterministic copy channel: marginal[0] = marginal[1] = identity
    # The independent channel equals the original, so MI = 0
    f = Channel(x, x.tensor(x), [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    mi = mutual_information(f, KL())
    assert mi == pytest.approx(0.0)


def test_mutual_information_channel_correlated():
    x = FiniteSpace([0, 1])
    # Non-deterministic channel where outputs are correlated
    # P(Y0,Y1 | X=0) = [0.8, 0.1, 0.1, 0.0], P(Y0,Y1 | X=1) = [0.0, 0.1, 0.1, 0.8]
    f = Channel(x, x.tensor(x), [[0.8, 0.0], [0.1, 0.1], [0.1, 0.1], [0.0, 0.8]])
    mi = mutual_information(f, KL())
    assert mi > 0.0


def test_mutual_information_wrong_factor_count_raises():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    with pytest.raises(ValueError, match="exactly two"):
        mutual_information(p, KL())


def test_mutual_information_channel_wrong_factors_raises():
    x = FiniteSpace([0, 1])
    f = Channel(x, x, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="exactly two"):
        mutual_information(f, KL())


def test_conditional_mutual_information_domain_mismatch_raises():
    a = FiniteSpace([0, 1])
    b = FiniteSpace(["a", "b"])
    xy = b.tensor(b)
    channel = Channel(a, xy, [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]])
    # Wrong source space
    with pytest.raises(ValueError, match="domain"):
        conditional_mutual_information(channel, Distribution(b, [0.5, 0.5]), KL())


def test_conditional_mutual_information_wrong_factors_raises():
    a = FiniteSpace([0, 1])
    b = FiniteSpace(["a", "b"])
    channel = Channel(a, b, [[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="two"):
        conditional_mutual_information(channel, Distribution(a, [0.5, 0.5]), KL())


def test_conditional_entropy_domain_mismatch_raises():
    a = FiniteSpace([0, 1])
    b = FiniteSpace(["a", "b"])
    source = Distribution(b, [0.5, 0.5])
    channel = Channel(a, b, [[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="domain"):
        conditional_entropy(channel, source, KL())


def test_renyi_entropy_edges():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    # order 0 → log(support size)
    assert renyi_entropy(p, 0) == pytest.approx(math.log(2))
    # order 1 → Shannon
    assert renyi_entropy(p, 1) == pytest.approx(shannon_entropy(p))
    # order inf → -log(max(p))
    assert renyi_entropy(p, float("inf")) == pytest.approx(-math.log(0.5))
    # deterministic distribution
    d = Distribution(x, [1.0, 0.0])
    assert renyi_entropy(d, 0) == pytest.approx(0.0)
    assert renyi_entropy(d, float("inf")) == pytest.approx(0.0)


def test_renyi_entropy_negative_order_raises():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    with pytest.raises(ValueError, match="order"):
        renyi_entropy(p, -1)


def test_entropy_channel():
    x = FiniteSpace([0, 1])
    det = Channel(x, x, np.eye(2))
    assert entropy(det, KL()) == pytest.approx(0.0)


def test_gini_simpson_uniform():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    assert gini_simpson(p) == pytest.approx(0.5)


# --- markov.py coverage ---


def test_joint_domain_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    p = Distribution(y, [0.5, 0.5])
    f = Channel(x, y, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="domain"):
        joint(p, f)


def test_is_independent_single_factor_raises():
    x = FiniteSpace([0, 1])
    p = Distribution(x, [0.5, 0.5])
    with pytest.raises(ValueError, match="product space"):
        is_independent(p)


def test_almost_sure_equal_type_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    f = Channel(x, y, [[1.0, 0.0], [0.0, 1.0]])
    g = Channel(x, x, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="same type"):
        almost_sure_equal(f, g, Distribution(x, [0.5, 0.5]))


def test_almost_sure_equal_source_mismatch_raises():
    x = FiniteSpace([0, 1])
    y = FiniteSpace(["a", "b"])
    f = Channel(x, y, [[1.0, 0.0], [0.0, 1.0]])
    g = Channel(x, y, [[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="domain"):
        almost_sure_equal(f, g, Distribution(y, [0.5, 0.5]))


# --- validation.py coverage ---


def test_probability_vector_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        probability_vector([0.5, 0.5, 0.0], 2)


def test_probability_vector_non_finite_raises():
    with pytest.raises(ValueError, match="finite"):
        probability_vector([float("nan"), 0.0], 2)


def test_probability_vector_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        probability_vector([-0.5, 1.5], 2)


def test_stochastic_matrix_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        stochastic_matrix([[0.5, 0.5]], 2, 1)


def test_stochastic_matrix_non_finite_raises():
    with pytest.raises(ValueError, match="finite"):
        stochastic_matrix([[float("nan")], [1.0]], 2, 1)


def test_stochastic_matrix_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        stochastic_matrix([[-0.5], [1.5]], 2, 1)


def test_stochastic_matrix_column_not_summing_to_one_raises():
    with pytest.raises(ValueError, match="sum to one"):
        stochastic_matrix([[0.3], [0.3]], 2, 1)
