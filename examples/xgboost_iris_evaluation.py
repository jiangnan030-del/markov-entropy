"""Train XGBoost on Iris and evaluate its probabilistic channel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from markov_entropy import (
    Channel,
    Distribution,
    FiniteSpace,
    conditional_entropy,
    entropy,
    joint,
    mutual_information,
)
from markov_entropy.divergences import KL


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Return top-label expected calibration error."""
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return error


def probability_channel(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[Distribution, Channel]:
    """Build P(model output | true class) by averaging predicted probabilities."""
    label_space = FiniteSpace(class_names)
    counts = np.bincount(y_true, minlength=len(class_names)).astype(float)
    source = Distribution(label_space, counts / counts.sum())
    matrix = np.column_stack(
        [
            probabilities[y_true == idx].mean(axis=0)
            for idx in range(len(class_names))
        ]
    )
    # Renormalise columns to mitigate floating-point drift from XGBoost
    matrix = matrix / matrix.sum(axis=0, keepdims=True)
    return source, Channel(label_space, label_space, matrix)


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[dict[str, float | int], Distribution, Channel]:
    """Combine standard classification metrics with information metrics."""
    predicted = probabilities.argmax(axis=1)
    source, channel = probability_channel(y_true, probabilities, class_names)
    predicted_distribution = channel.apply(source)
    true_predicted_joint = joint(source, channel)

    true_entropy = entropy(source, KL())
    information = mutual_information(true_predicted_joint, KL())
    brier = float(
        np.mean(np.sum((probabilities - np.eye(len(class_names))[y_true]) ** 2, axis=1))
    )
    metrics: dict[str, float | int] = {
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "multiclass_log_loss": float(
            log_loss(y_true, probabilities, labels=np.arange(len(class_names)))
        ),
        "multiclass_brier_score": brier,
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "true_label_entropy_nats": true_entropy,
        "predicted_probability_entropy_nats": entropy(predicted_distribution, KL()),
        "conditional_prediction_entropy_nats": conditional_entropy(channel, source, KL()),
        "mutual_information_nats": information,
        "information_efficiency": information / true_entropy if true_entropy else 0.0,
    }
    return metrics, source, channel


def write_artifacts(
    output_dir: Path,
    metrics: dict[str, float | int],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
    source: Distribution,
    channel: Channel,
) -> None:
    """Write reproducible metrics, channel, and prediction artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output_dir / "probability_channel.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_class", "class_prior", *(f"p_{name}" for name in class_names)])
        for class_index, class_name in enumerate(class_names):
            writer.writerow(
                [
                    class_name,
                    float(source.probabilities[class_index]),
                    *(float(value) for value in channel.matrix[:, class_index]),
                ]
            )

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["true_class", "predicted_class", "confidence", *(f"p_{name}" for name in class_names)]
        )
        for row_index in range(len(y_true)):
            predicted_index = int(probabilities[row_index].argmax())
            writer.writerow(
                [
                    class_names[int(y_true[row_index])],
                    class_names[predicted_index],
                    float(probabilities[row_index, predicted_index]),
                    *(float(value) for value in probabilities[row_index]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/xgboost-iris"),
        help="directory for JSON and CSV artifacts",
    )
    args = parser.parse_args()

    iris = load_iris()
    features = np.asarray(iris.data, dtype=float)
    targets = np.asarray(iris.target, dtype=int)
    class_names = tuple(str(name) for name in iris.target_names)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.30,
        random_state=42,
        stratify=targets,
    )

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_estimators=100,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(x_train, y_train)
    probabilities = np.asarray(classifier.predict_proba(x_test), dtype=float)
    # Normalise to mitigate XGBoost floating-point drift so rows sum to one
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    metrics, source, channel = evaluate(y_test, probabilities, class_names)
    write_artifacts(
        args.output_dir,
        metrics,
        y_test,
        probabilities,
        class_names,
        source,
        channel,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("\nProbability channel P(model output | true class):")
    print(np.array2string(channel.matrix, precision=4, suppress_small=True))
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
