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
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from tradon.backtest import BacktestResult, run_backtest
from tradon.data.base import Candle
from tradon.ml.features import FEATURE_NAMES, MIN_CANDLES, extract_features
from tradon.risk.manager import RiskManager
from tradon.strategy.base import Signal, Strategy


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
    # (feature name, importance) pairs, sorted highest first. Makes the
    # model at least partly inspectable instead of a pure black box — if
    # "hour_sin"/"dow_sin" dominate, that's a very different (and much
    # more suspect) model than one driven by rsi_14/macd_hist.
    feature_importances: tuple[tuple[str, float], ...]


def _normalized_permutation_importances(
    model: Any, x_test: list[list[float]], y_test: list[int]
) -> list[tuple[str, float]]:
    """HistGradientBoostingClassifier has no built-in `feature_importances_`
    (unlike tree-ensemble models such as GradientBoostingClassifier), so
    importance is measured instead by how much held-out test accuracy
    drops when a feature's values are shuffled — permutation_importance,
    evaluated on the test split so it reflects generalization, not fit.
    A shuffled-but-useless feature can score slightly negative (shuffling
    it occasionally helps by luck); those are clipped to 0 so every value
    sums to 1 like the old feature_importances_-based report did.
    """
    result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=0)
    clipped = [max(0.0, float(v)) for v in result.importances_mean]
    total = sum(clipped)
    normalized = [v / total if total > 0 else 0.0 for v in clipped]
    return list(zip(FEATURE_NAMES, normalized, strict=True))


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

    # Regularized (shallow, slow, L2-penalized, early-stopped) rather than
    # scikit-learn's unregularized defaults — a plain GradientBoostingClassifier
    # memorized its training window (98% train accuracy) and lost badly
    # out-of-sample on a live walk-forward comparison (chained Sharpe
    # -6.45); this config still doesn't turn a profit out-of-sample either,
    # but overfits far less and had the best (least negative) Sharpe of
    # four candidates tried. See README's ML section for the numbers.
    model = HistGradientBoostingClassifier(
        random_state=0, max_depth=3, learning_rate=0.05, l2_regularization=1.0, early_stopping=True
    )
    model.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(x_train))
    test_accuracy = accuracy_score(y_test, model.predict(x_test))
    majority_class = max(set(y_test), key=y_test.count)
    baseline_accuracy = y_test.count(majority_class) / len(y_test)
    importances = tuple(sorted(_normalized_permutation_importances(model, x_test, y_test), key=lambda p: p[1], reverse=True))

    report = TrainingReport(
        samples=total_samples,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        test_baseline_accuracy=baseline_accuracy,
        feature_importances=tuple(importances),
    )
    return model, report


class _WalkForwardMLStrategy(Strategy):
    """Wraps an in-memory, already-fitted model as a Strategy so each
    walk-forward fold can be traded through the exact same run_backtest()
    every indicator strategy uses — MLStrategy itself only knows how to
    load a model from a saved path, not hold one that was just fit.
    """

    name = "ml"

    def __init__(self, model: Any, up_index: int, long_threshold: float, short_threshold: float) -> None:
        super().__init__({})
        self._model = model
        self._up_index = up_index
        self._long_threshold = long_threshold
        self._short_threshold = short_threshold

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        from tradon.strategy.plugins.ml_strategy import predict_signal

        return predict_signal(self._model, self._up_index, self._long_threshold, self._short_threshold, candles)


@dataclass(frozen=True, slots=True)
class MLWalkForwardFold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    training_report: TrainingReport
    out_of_sample: BacktestResult


def walk_forward_train(
    candles: list[Candle],
    horizon: int = 5,
    n_splits: int = 5,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    starting_balance: Decimal = Decimal(10000),
    risk_manager: RiskManager | None = None,
) -> list[MLWalkForwardFold]:
    """ML's analogue of tradon.backtest.walk_forward(): optimize-params and
    a single static train_model() call share the same flaw — both grade a
    choice (params, or a fitted model) on the very data used to make that
    choice, which overstates how it would do going forward. This instead
    splits `candles` into `n_splits` consecutive folds and, for each one,
    RETRAINS from scratch on everything before it, then trades only that
    model on the fold after (data it never saw during training) — chaining
    balance across folds (see walk_forward_summary() in backtest.py for
    why that's valid here: the test windows are consecutive and
    non-overlapping by construction). A model that looked great in
    train_model()'s single static split can still fall apart here if it
    doesn't hold up when periodically refit on fresh data, which is
    exactly the scenario live trading would actually put it through.
    """
    from tradon.strategy.plugins.ml_strategy import model_up_index

    if n_splits < 2:
        raise ValueError("walk_forward_train needs at least 2 splits (1 train, 1 test)")
    risk_manager = risk_manager or RiskManager()
    boundaries = [round(len(candles) * k / n_splits) for k in range(n_splits + 1)]

    folds: list[MLWalkForwardFold] = []
    balance = starting_balance
    for i in range(n_splits - 1):
        train_candles = candles[boundaries[0] : boundaries[i + 1]]
        test_candles = candles[boundaries[i + 1] : boundaries[i + 2]]
        if len(test_candles) < 2:
            continue

        try:
            model, report = train_model(train_candles, horizon=horizon, test_fraction=0.2)
        except ValueError:
            # Not enough/varied-enough data in this fold's training window
            # yet (common for the earliest folds) — skip it, not fatal.
            continue

        strategy = _WalkForwardMLStrategy(model, model_up_index(model), long_threshold, short_threshold)
        out_of_sample = run_backtest(test_candles, strategy, risk_manager, balance)
        folds.append(
            MLWalkForwardFold(
                train_start=train_candles[0].timestamp,
                train_end=train_candles[-1].timestamp,
                test_start=test_candles[0].timestamp,
                test_end=test_candles[-1].timestamp,
                training_report=report,
                out_of_sample=out_of_sample,
            )
        )
        balance = out_of_sample.ending_balance

    return folds


def walk_forward_train_summary(folds: list[MLWalkForwardFold]) -> BacktestResult:
    """Chain every fold's out-of-sample trading result into one equity
    curve — same reasoning as backtest.py's walk_forward_summary().
    """
    if not folds:
        return BacktestResult()
    trades = [t for fold in folds for t in fold.out_of_sample.trades]
    return BacktestResult(
        trades=trades,
        starting_balance=folds[0].out_of_sample.starting_balance,
        ending_balance=folds[-1].out_of_sample.ending_balance,
    )


def save_model(model: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
