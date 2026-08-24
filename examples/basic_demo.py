"""Run a small reproducible demonstration."""

from markov_entropy import Distribution, FiniteSpace, entropy, gini_simpson, mutual_information
from markov_entropy.divergences import KL, Renyi, TotalVariation

X = FiniteSpace(["0", "1"])
p = Distribution(X, [0.25, 0.75])
correlated = Distribution(X.tensor(X), [0.5, 0.0, 0.0, 0.5])

print(f"Shannon entropy: {entropy(p, KL()):.8f} nats")
print(f"Rényi-induced entropy (alpha=0.5): {entropy(p, Renyi(0.5)):.8f}")
print(f"Gini-Simpson via TV: {entropy(p, TotalVariation()):.8f}")
print(f"Gini-Simpson direct: {gini_simpson(p):.8f}")
print(f"Correlated mutual information: {mutual_information(correlated, KL()):.8f} nats")
