# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog, and versions follow Semantic Versioning.

## [Unreleased]

### Added

- Optional DisCoPy 1.2.2 adapter for symbolic finite Markov string diagrams.
- Symbolic channel, state, copy, discard, composition, and tensor constructions.
- Diagram pairs for entropy, channel entropy, mutual information, and conditional-output independence.
- Executable DisCoPy notebook, adapter documentation, and CI-tested optional dependency.
- API and numerical-conventions documentation.
- Property tests for KL chain rule, enriched composition/tensor bounds, copy/discard laws, and conditional information identities.
- Package build and wheel-install smoke test in CI.
- Contribution guidance and GitHub issue/PR templates.

## [0.1.0] - 2026-08-26

### Added

- Immutable finite spaces, distributions, and column-stochastic channels.
- Identity, composition, tensor, copy, discard, joint, marginal, determinism, independence, and almost-sure equality operations.
- KL, Rényi, and total-variation divergences with distribution and channel lifting.
- Mutual information, conditional mutual information, entropy, conditional entropy, Shannon entropy, Rényi entropy, and Gini–Simpson index.
- Log-domain Rényi evaluation with stable behavior near `alpha = 1` and extreme probabilities.
- BSC, BEC, XOR, data-processing, tensor-additivity, and Hypothesis property tests.
- Four executable notebooks, formula map, README example test, uv lockfile, and Python 3.11–3.13 CI.

[Unreleased]: https://github.com/jiangnan030-del/markov-entropy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jiangnan030-del/markov-entropy/releases/tag/v0.1.0
