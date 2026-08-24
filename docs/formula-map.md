# Formula-to-code map

| Paper | Implementation |
|---|---|
| Definition 2.4, KL divergence | `KL` |
| Definition 2.5, enriched composition/tensor bounds | property tests |
| Definition 2.9, channel KL as maximum over inputs | `channel_divergence` |
| Definition 2.16, Rényi divergence | `Renyi` |
| Definition 2.20, total variation | `TotalVariation` |
| Equation (21), almost-sure equality | `almost_sure_equal` |
| Definition 3.1, mutual information | `mutual_information` |
| Definition 4.1, entropy from non-determinism | `entropy` |
| Section 4.2.3, Gini-Simpson index | `gini_simpson` |

The implementation uses natural logarithms and finite alphabets. It does not identify categorical entropy on general measurable spaces with differential entropy.
