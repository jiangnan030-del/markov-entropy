"""Verify that the code snippet in README.md runs correctly."""


from markov_entropy import Distribution, FiniteSpace
from markov_entropy.divergences import KL, Renyi, TotalVariation
from markov_entropy.information import entropy


def test_readme_quick_start():
    """Run the exact code from the README Quick Start section."""
    X = FiniteSpace(["0", "1"])
    p = Distribution(X, [0.25, 0.75])

    shannon = entropy(p, KL())
    renyi = entropy(p, Renyi(alpha=1.0))
    gini_simpson = entropy(p, TotalVariation())

    assert shannon > 0
    assert renyi == shannon
    assert gini_simpson == 1 - (0.25**2 + 0.75**2)
