"""Verify that the code snippet in README.md runs correctly."""

from markov_entropy import Distribution, FiniteSpace
from markov_entropy.divergences import KL
from markov_entropy.information import entropy


def test_readme_quick_start():
    """Run the exact code from the README Quick Start section."""
    X = FiniteSpace(["0", "1"])
    p = Distribution(X, [0.25, 0.75])
    assert entropy(p, KL()) > 0
