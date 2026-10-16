"""Offline training for the ML strategy (`tradon train-model` CLI command).

Kept separate from MLStrategy itself: fitting a classifier needs a full
historical dataset and scikit-learn's fit(), which don't belong in the
same lightweight on_candles() call path every other strategy uses. Train
here, save to disk, and MLStrategy just loads the result and calls
predict_proba() — the same "train offline, infer online" split any real
ML deployment uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from tradon.data.base import Candle
from tradon.ml.features import MIN_CANDLES, extract_features


@dataclass(frozen=True, slots=True)
class TrainingReport:
    samples: int
    train_accuracy: float
    test_accuracy: float
    # Accuracy of "always predict the test period's more common outcome" —
    # the bar the model actually has to clear to be worth anything. A model
    # that can't beat this has learned nothing the class balance didn't
    # already give it for free.
    test_baseline_accuracy: float


def build_dataset(candles: list[Candle], horizon: int) -> tuple[list[list[float]], list[int]]:
    """One row per bar: features as of that bar, label = whether the close
    `horizon` bars later ended up higher (1) than the close at that bar
    (0 otherwise). The last `horizon` bars can't be labeled yet (their
    future isn't in `candles`) and are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    features: list[list[float]] = []
    labels: list[int] = []
    for i in range(MIN_CANDLES, len(candles) - horizon):
        row = extract_features(candles[: i + 1])
        if row is None:
            continue
        features.append(row)
        labels.append(1 if candles[i + horizon].close > candles[i].close else 0)
    return features, labels


def train_model(
    candles: list[Candle],
    horizon: int = 5,
    test_fraction: float = 0.25,
) -> tuple[Any, TrainingReport]:
    """Fits a classifier on ONE symbol's history. Thin convenience wrapper
    around train_model_multi() — see that function for the actual fitting
    logic and for pooling several symbols instead of just one.
    """
    return train_model_multi({"symbol": candles}, horizon, test_fraction)


def train_model_multi(
    candles_by_symbol: dict[str, list[Candle]],
    horizon: int = 5,
    test_fraction: float = 0.25,
) -> tuple[Any, TrainingReport]:
    """Fits a gradient-boosted classifier to predict "will price be higher
    `horizon` bars from now than it is on this bar", pooling rows from
    every symbol in `candles_by_symbol` so the model learns patterns that
    hold across instruments instead of overfitting to one symbol's
    specific behavior (a cleaner trend-vs-chop distinction, say, rather
    than "BTCUSDT's quirks in this particular window").

    The chronological train/test split (see train_model()'s docstring for
    why it must be chronological, not shuffled) is done PER SYMBOL first —
    each symbol's own earliest `1 - test_fraction` of bars go to train,
    its own latest `test_fraction` go to test — and only then pooled
    across symbols. Splitting after pooling would be wrong: two different
    symbols' bars at "the same relative position" aren't the same point
    in time, so a cut across the merged set wouldn't actually guarantee
    every training row precedes every test row the way a real
    look-ahead-safe split requires.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not candles_by_symbol:
        raise ValueError("Need at least one symbol's candles to train on")

    x_train: list[list[float]] = []
    y_train: list[int] = []
    x_test: list[list[float]] = []
    y_test: list[int] = []
    total_samples = 0

    for candles in candles_by_symbol.values():
        features, labels = build_dataset(candles, horizon)
        total_samples += len(features)
        split = int(len(features) * (1 - test_fraction))
        x_train.extend(features[:split])
        y_train.extend(labels[:split])
        x_test.extend(features[split:])
        y_test.extend(labels[split:])

    if total_samples < 50:
        raise ValueError(f"Only {total_samples} labeled samples — need at least 50 to train anything meaningful")
    if len(set(y_train)) < 2:
        raise ValueError("Training split has only one outcome (all up or all down) — need more/more varied history")
    if len(set(y_test)) < 2:
        raise ValueError("Test split has only one outcome — try a smaller --test-fraction or more --limit")

    model = GradientBoostingClassifier(random_state=0)
    model.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(x_train))
    test_accuracy = accuracy_score(y_test, model.predict(x_test))
    majority_class = max(set(y_test), key=y_test.count)
    baseline_accuracy = y_test.count(majority_class) / len(y_test)

    report = TrainingReport(
        samples=total_samples,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        test_baseline_accuracy=baseline_accuracy,
    )
    return model, report


def save_model(model: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
