import numpy as np
from hypothesis import given, settings, strategies as st

from markov_entropy import (
    Channel,
    Distribution,
    FiniteSpace,
    conditional_entropy,
    conditional_mutual_information,
    copy,
    discard,
    entropy,
    identity,
    joint,
    mutual_information,
)
from markov_entropy.divergences import (
    KL,
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
    column_values = draw(
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
    matrix = np.asarray(column_values, dtype=np.float64).T
    return matrix / matrix.sum(axis=0, keepdims=True)


def swap_channel(left: FiniteSpace, right: FiniteSpace) -> Channel:
    domain = left.tensor(right)
    codomain = right.tensor(left)
    matrix = np.zeros((len(codomain), len(domain)))
    target_indices = {label: index for index, label in enumerate(codomain.labels)}
    for source_index, (left_label, right_label) in enumerate(domain.labels):
        matrix[target_indices[(right_label, left_label)], source_index] = 1.0
    return Channel(domain, codomain, matrix)


@settings(max_examples=35, deadline=None)
@given(
    probability_values(size=2),
    probability_values(size=2),
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=3, columns=2),
)
def test_kl_chain_rule(source_values, reference_values, first_matrix, second_matrix):
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b", "c"))
    source = Distribution(x, source_values)
    reference = Distribution(x, reference_values)
    first = Channel(x, y, first_matrix)
    second = Channel(x, y, second_matrix)

    lhs = distribution_divergence(joint(source, first), joint(reference, second), KL())
    conditional = sum(
        source.probabilities[index] * KL()(first.matrix[:, index], second.matrix[:, index])
        for index in range(len(x))
    )
    rhs = distribution_divergence(source, reference, KL()) + conditional
    assert np.isclose(lhs, rhs, atol=1e-10, rtol=1e-10)


@settings(max_examples=35, deadline=None)
@given(
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=2, columns=3),
    stochastic_matrices(rows=2, columns=3),
)
def test_channel_composition_subadditivity(
    first_matrix,
    second_matrix,
    post_first_matrix,
    post_second_matrix,
):
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b", "c"))
    z = FiniteSpace(("left", "right"))
    first = Channel(x, y, first_matrix)
    second = Channel(x, y, second_matrix)
    post_first = Channel(y, z, post_first_matrix)
    post_second = Channel(y, z, post_second_matrix)

    for divergence in (KL(), TotalVariation()):
        lhs = channel_divergence(
            post_first.compose(first),
            post_second.compose(second),
            divergence,
        )
        rhs = channel_divergence(first, second, divergence) + channel_divergence(
            post_first,
            post_second,
            divergence,
        )
        assert lhs <= rhs + 1e-10


@settings(max_examples=35, deadline=None)
@given(
    stochastic_matrices(rows=2, columns=2),
    stochastic_matrices(rows=2, columns=2),
    stochastic_matrices(rows=3, columns=2),
    stochastic_matrices(rows=3, columns=2),
)
def test_channel_tensor_bounds(first_matrix, second_matrix, left_matrix, right_matrix):
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b"))
    a = FiniteSpace(("u", "v"))
    b = FiniteSpace((0, 1, 2))
    first = Channel(x, y, first_matrix)
    second = Channel(x, y, second_matrix)
    left = Channel(a, b, left_matrix)
    right = Channel(a, b, right_matrix)

    kl_lhs = channel_divergence(first.tensor(left), second.tensor(right), KL())
    kl_rhs = channel_divergence(first, second, KL()) + channel_divergence(left, right, KL())
    assert np.isclose(kl_lhs, kl_rhs, atol=1e-10, rtol=1e-10)

    tv_lhs = channel_divergence(
        first.tensor(left),
        second.tensor(right),
        TotalVariation(),
    )
    tv_rhs = channel_divergence(first, second, TotalVariation()) + channel_divergence(
        left,
        right,
        TotalVariation(),
    )
    assert tv_lhs <= tv_rhs + 1e-10


def test_copy_is_coassociative_and_commutative():
    x = FiniteSpace((0, 1, 2))
    left = copy(x).tensor(identity(x)).compose(copy(x))
    right = identity(x).tensor(copy(x)).compose(copy(x))
    assert left.domain == right.domain
    assert left.codomain == right.codomain
    assert np.array_equal(left.matrix, right.matrix)

    swapped = swap_channel(x, x).compose(copy(x))
    assert np.array_equal(swapped.matrix, copy(x).matrix)


def test_discard_is_a_copy_counit_and_tensor_unit_is_numerically_strict():
    x = FiniteSpace((0, 1, 2))
    left_counit = discard(x).tensor(identity(x)).compose(copy(x))
    right_counit = identity(x).tensor(discard(x)).compose(copy(x))
    expected = identity(x).matrix
    assert np.array_equal(left_counit.matrix, expected)
    assert np.array_equal(right_counit.matrix, expected)

    channel = Channel(x, x, np.eye(len(x)))
    assert np.array_equal(channel.tensor(identity(discard(x).codomain)).matrix, channel.matrix)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=1), min_size=3, max_size=3))
def test_copy_is_natural_for_deterministic_channels(targets):
    x = FiniteSpace((0, 1, 2))
    y = FiniteSpace(("left", "right"))
    matrix = np.zeros((len(y), len(x)))
    for source_index, target_index in enumerate(targets):
        matrix[target_index, source_index] = 1.0
    channel = Channel(x, y, matrix)

    left = copy(y).compose(channel)
    right = channel.tensor(channel).compose(copy(x))
    assert np.array_equal(left.matrix, right.matrix)


@settings(max_examples=30, deadline=None)
@given(stochastic_matrices(rows=2, columns=3))
def test_discard_is_natural_for_all_channels(matrix):
    x = FiniteSpace((0, 1, 2))
    y = FiniteSpace(("left", "right"))
    channel = Channel(x, y, matrix)
    assert np.array_equal(discard(y).compose(channel).matrix, discard(x).matrix)


@settings(max_examples=35, deadline=None)
@given(probability_values(size=2), stochastic_matrices(rows=3, columns=2))
def test_conditional_entropy_is_expected_fibre_entropy(source_values, matrix):
    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b", "c"))
    source = Distribution(x, source_values)
    channel = Channel(x, y, matrix)
    expected = sum(
        source.probabilities[index]
        * entropy(Distribution(y, channel.matrix[:, index]), KL())
        for index in range(len(x))
    )
    assert conditional_entropy(channel, source, KL()) == np.testing.assert_allclose(
        conditional_entropy(channel, source, KL()), expected, atol=1e-10, rtol=1e-10
    )


@settings(max_examples=35, deadline=None)
@given(probability_values(size=2), stochastic_matrices(rows=4, columns=2))
def test_conditional_mutual_information_is_expected_fibre_information(
    source_values,
    matrix,
):
    a = FiniteSpace((0, 1))
    x = FiniteSpace(("x0", "x1"))
    y = FiniteSpace(("y0", "y1"))
    source = Distribution(a, source_values)
    channel = Channel(a, x.tensor(y), matrix)
    expected = sum(
        source.probabilities[index]
        * mutual_information(Distribution(channel.codomain, channel.matrix[:, index]), KL())
        for index in range(len(a))
    )
    np.testing.assert_allclose(
        conditional_mutual_information(channel, source, KL()),
        expected,
        atol=1e-10,
        rtol=1e-10,
    )
