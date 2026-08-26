# Formula-to-code map

| Paper construction | Implementation | Regression coverage |
|---|---|---|
| Definition 2.4, KL divergence | `KL` | edge cases, chain rule, data processing |
| Definition 2.5, enriched composition/tensor bounds | `channel_divergence` | composition and tensor property tests |
| Definition 2.9, channel KL as maximum over inputs | `channel_divergence` | channel divergence tests |
| Definition 2.16, Rényi divergence | `Renyi` | special orders, extreme probabilities, `α → 1` |
| Definition 2.20, total variation | `TotalVariation` | edge cases and contraction tests |
| Equation (21), almost-sure equality | `almost_sure_equal` | support-sensitive equality tests |
| Markov copy/discard structure | `copy`, `discard` | coassociativity, commutativity, counit, naturality |
| Definition 3.1, mutual information | `mutual_information` | independence, BSC, BEC, data processing |
| Conditional mutual information | `conditional_mutual_information` | XOR and fibre-expectation identities |
| Definition 4.1, entropy from non-determinism | `entropy` | Shannon, Rényi, deterministic processing |
| Conditional entropy | `conditional_entropy` | source-weighted fibre entropy |
| Section 4.2.3, Gini–Simpson index | `gini_simpson` | total-variation recovery |

The implementation uses natural logarithms and finite alphabets. It does not identify categorical entropy on general measurable spaces with differential entropy.

See also [API reference](api.md) and [Numerical conventions](conventions.md).
