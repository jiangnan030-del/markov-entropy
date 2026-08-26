# XGBoost Iris classifier evaluation

This example turns a multiclass classifier into a finite probabilistic channel and evaluates it with both conventional machine-learning metrics and `markov-entropy` information quantities.

## Run

From the repository root:

```bash
uv run --with "scikit-learn>=1.5" --with "xgboost>=2.1" \
  python examples/xgboost_iris_evaluation.py
```

Dependencies are supplied with `uv --with`, so XGBoost and scikit-learn do not become runtime dependencies of the core library.

The run uses a deterministic, stratified 70/30 train/test split and writes:

- `build/xgboost-iris/metrics.json`
- `build/xgboost-iris/probability_channel.csv`
- `build/xgboost-iris/test_predictions.csv`

## Model

The example trains `XGBClassifier` with the `multi:softprob` objective. It retains the full class-probability vector for every test observation instead of reducing evaluation to hard labels.

For each true class `y`, the example averages the probability vectors assigned by the model:

```text
P(model output = k | true class = y)
```

The resulting column-stochastic matrix is a `markov_entropy.Channel`. The empirical test-set class frequencies form the source `Distribution`.

This is a soft probability channel. It differs from a normalized hard-label confusion matrix because every prediction contributes probability mass to all three classes.

## Reported metrics

### Standard classifier metrics

- Accuracy
- Multiclass logarithmic loss
- Multiclass Brier score
- Top-label expected calibration error

### Information-theoretic metrics

- True-label entropy `H(Y)`
- Entropy of the marginal predicted probability distribution `H(Ŷ)`
- Conditional prediction entropy `H(Ŷ | Y)`
- Mutual information `I(Y; Ŷ)`
- Information efficiency `I(Y; Ŷ) / H(Y)`

All entropy and mutual-information values use natural logarithms and are reported in nats.

## Interpretation

- Higher accuracy and lower log loss, Brier score, and calibration error are better.
- Higher `I(Y; Ŷ)` means the model-output channel preserves more information about the true species.
- Lower `H(Ŷ | Y)` means probability outputs are more concentrated and consistent within each true class.
- Information efficiency near one means that most test-label uncertainty is recoverable from the model's probabilistic output.

These quantities should be interpreted together. A highly confident but miscalibrated model can have concentrated outputs while still producing poor log loss or Brier score.

## Scope

The Iris dataset is intentionally small and is suitable for a reproducible demonstration, not a production benchmark. For operational evaluation, use repeated cross-validation or an untouched external test set, confidence intervals, class-specific calibration, and drift monitoring.
