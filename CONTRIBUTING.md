# Contributing

Contributions are welcome, especially mathematical regression tests, numerical-stability improvements, documentation, and carefully scoped categorical interfaces.

## Development setup

```bash
git clone https://github.com/jiangnan030-del/markov-entropy.git
cd markov-entropy
uv sync --all-extras
```

## Required checks

Run the same checks as CI before opening a pull request:

```bash
uv run ruff check .
uv run mypy src/markov_entropy
uv run pytest
uv run pytest --nbval examples/ -p no:cacheprovider --override-ini="addopts="
uv run pytest tests/test_readme_doctest.py --override-ini="addopts="
uv build
```

## Mathematical changes

A change to a formula or numerical convention should include:

1. the mathematical definition and assumptions;
2. at least one direct regression test;
3. boundary cases involving zeros or infinite values where relevant;
4. a reference to the corresponding paper section or standard result;
5. documentation of any tolerance or approximation.

## Pull requests

Keep changes focused. Explain the mathematical meaning, implementation choice, tests, and compatibility impact. Do not claim support for general measurable spaces unless the assumptions and approximation error are explicit.
