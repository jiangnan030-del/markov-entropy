# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog, and versions follow Semantic Versioning.

## [Unreleased]

### Added

- Experimental density divergences with a user-supplied common-base-measure integrator.
- Finite-partition divergence lower-bound sequences with monotonicity diagnostics.
- Monte Carlo expectation and KL estimators with standard errors and confidence intervals.
- Explicit documentation and notebook for assumptions, uncertainty, and non-goals.

## [0.2.0] - 2026-08-26

### Added

- Optional DisCoPy 1.2.2 adapter for symbolic finite Markov string diagrams.
- Symbolic state and channel boxes, copy, discard, composition, and tensor diagrams.
- Diagram comparisons for state entropy, channel entropy, mutual information, and conditional-output independence.
- Reproducible SVG gallery renderer and downloadable CI gallery artifact.
- Expanded visual notebook linking categorical diagrams to numerical entropy and mutual information.
- DisCoPy adapter documentation and diagram gallery.

### Changed

- Project status advanced from alpha to beta.
- Package description and documentation links now include string-diagram support.

## [0.1.0] - 2026-08-26

### Added

- Immutable finite spaces, distributions, and column-stochastic channels.
- Identity, composition, tensor, copy, discard, joint, marginal, determinism, independence, and almost-sure equality operations.
- KL, Rényi, and total-variation divergences with distribution and channel lifting.
- Mutual information, conditional mutual information, entropy, conditional entropy, Shannon entropy, Rényi entropy, and Gini–Simpson index.
- Log-domain Rényi evaluation with stable behavior near `alpha = 1` and extreme probabilities.
- BSC, BEC, XOR, data-processing, tensor-additivity, and Hypothesis property tests.
- Four executable notebooks, formula map, README example test, uv lockfile, and Python 3.11–3.13 CI.
- API and numerical-conventions documentation.
- Property tests for KL chain rule, enriched composition/tensor bounds, copy/discard laws, and conditional information identities.
- Package build and wheel-install smoke test in CI.
- Contribution guidance and GitHub issue/PR templates.

[Unreleased]: https://github.com/jiangnan030-del/markov-entropy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jiangnan030-del/markov-entropy/releases/tag/v0.2.0
[0.1.0]: https://github.com/jiangnan030-del/markov-entropy/releases/tag/v0.1.0
