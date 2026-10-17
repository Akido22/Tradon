"""Tests for the optional ML strategy (pip install -e .[ml]).

Skipped entirely when scikit-learn/joblib aren't installed, same as how
MetaTrader5-specific tests would be skipped on a machine without that
package — ML is an optional extra, not a hard dependency of the project.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from tradon.data.base import Candle  # noqa: E402
from tradon.ml.features import FEATURE_NAMES, MIN_CANDLES, extract_features  # noqa: E402
from tradon.ml.train import (  # noqa: E402
    build_dataset,
    load_model,
    save_model,
    train_model,
    train_model_multi,
    walk_forward_train,
    walk_forward_train_summary,
)
from tradon.strategy.plugins.ml_strategy import MLStrategy  # noqa: E402


def make_candles(closes: list[float], spread: float = 0.05) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = []
    prev_close = closes[0]
    for i, close in enumerate(closes):
        open_ = prev_close
        high = max(open_, close) + spread
        low = min(open_, close) - spread
        candles.append(
            Candle(
                symbol="TEST",
                timeframe="1h",
                open=Decimal(str(open_)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                volume=Decimal(100),
                timestamp=start + timedelta(hours=i),
                source="test",
            )
        )
        prev_close = close
    return candles


def make_oscillating_closes(n: int) -> list[float]:
    return [100.0 + 5 * math.sin(i / 6) + 0.03 * i for i in range(n)]


def make_cyclical_closes(n_cycles: int, half_len: int = 15, step: float = 1.0) -> list[float]:
    """A repeating sawtooth: `half_len` bars rising by `step` each, then
    `half_len` bars falling by `step` each, repeated `n_cycles` times.
    Strongly and consistently predictable from recent momentum (so a
    classifier should learn it cleanly), but — unlike a monotonic trend —
    both up and down labels actually occur, which train_model requires
    (it rejects an all-one-class split).
    """
    closes = [100.0]
    for _ in range(n_cycles):
        for _ in range(half_len):
            closes.append(closes[-1] + step)
        for _ in range(half_len):
            closes.append(closes[-1] - step)
    return closes


def test_extract_features_none_before_warmup() -> None:
    candles = make_candles([100.0] * (MIN_CANDLES - 1))
    assert extract_features(candles) is None


def test_extract_features_returns_one_value_per_feature_name() -> None:
    candles = make_candles(make_oscillating_closes(MIN_CANDLES + 5))
    features = extract_features(candles)
    assert features is not None
    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features)


def test_feature_names_include_volume_volatility_and_time_features() -> None:
    assert {"volume_ratio", "volatility_20", "hour_sin", "hour_cos", "dow_sin", "dow_cos"} <= set(FEATURE_NAMES)


def test_extract_features_cyclical_time_encoding_is_bounded() -> None:
    candles = make_candles(make_oscillating_closes(MIN_CANDLES + 5))
    features = extract_features(candles)
    assert features is not None
    by_name = dict(zip(FEATURE_NAMES, features, strict=True))
    for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert -1.0 <= by_name[name] <= 1.0


def test_build_dataset_label_matches_future_direction() -> None:
    candles = make_candles(make_oscillating_closes(200))
    horizon = 5
    features, labels = build_dataset(candles, horizon)

    assert len(features) == len(labels)
    assert len(features) > 0
    # Reconstruct the label for the first row by hand and check it matches.
    first_row_index = MIN_CANDLES
    expected = 1 if candles[first_row_index + horizon].close > candles[first_row_index].close else 0
    assert labels[0] == expected


def test_build_dataset_rejects_zero_horizon() -> None:
    candles = make_candles(make_oscillating_closes(100))
    with pytest.raises(ValueError):
        build_dataset(candles, horizon=0)


def test_train_model_raises_on_too_little_data() -> None:
    candles = make_candles(make_oscillating_closes(MIN_CANDLES + 5))
    with pytest.raises(ValueError):
        train_model(candles, horizon=5)


def test_train_model_learns_a_predictable_cycle() -> None:
    closes = make_cyclical_closes(n_cycles=20, half_len=15)  # 600 bars, evenly split up/down
    candles = make_candles(closes)

    model, report = train_model(candles, horizon=5, test_fraction=0.2)

    assert report.samples > 50
    # The real bar: did it learn the pattern, not just the class balance.
    assert report.test_accuracy > report.test_baseline_accuracy
    assert report.train_accuracy > 0.6

    probability_up = model.predict_proba([extract_features(candles)])[0][list(model.classes_).index(1)]
    assert 0 <= probability_up <= 1


def test_train_model_multi_pools_several_symbols() -> None:
    candles_a = make_candles(make_cyclical_closes(n_cycles=20, half_len=15))
    candles_b = make_candles(make_cyclical_closes(n_cycles=20, half_len=15, step=0.5))

    model, report = train_model_multi({"A": candles_a, "B": candles_b}, horizon=5, test_fraction=0.2)

    assert report.test_accuracy > report.test_baseline_accuracy
    # Both symbols' rows actually went in — not just one of them.
    _, report_a = train_model(candles_a, horizon=5, test_fraction=0.2)
    _, report_b = train_model(candles_b, horizon=5, test_fraction=0.2)
    assert report.samples == report_a.samples + report_b.samples


def test_train_model_multi_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        train_model_multi({}, horizon=5)


def test_train_model_multi_raises_on_too_little_combined_data() -> None:
    tiny_a = make_candles(make_oscillating_closes(MIN_CANDLES + 2))
    tiny_b = make_candles(make_oscillating_closes(MIN_CANDLES + 2))
    with pytest.raises(ValueError):
        train_model_multi({"A": tiny_a, "B": tiny_b}, horizon=5)


def test_ml_strategy_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        MLStrategy({})


def test_ml_strategy_rejects_missing_model_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="No trained model"):
        MLStrategy({"model_path": str(tmp_path / "does_not_exist.joblib")})


def test_ml_strategy_produces_both_directions_on_a_predictable_cycle(tmp_path) -> None:
    closes = make_cyclical_closes(n_cycles=20, half_len=15)
    candles = make_candles(closes)
    model, report = train_model(candles, horizon=5, test_fraction=0.2)
    assert report.test_accuracy > report.test_baseline_accuracy  # sanity: it actually learned the cycle
    model_path = str(tmp_path / "model.joblib")
    save_model(model, model_path)

    strategy = MLStrategy({"model_path": model_path, "long_threshold": 0.55, "short_threshold": 0.45})

    # Sample the strategy at many points across the cycle rather than
    # asserting one exact direction at one exact bar — a classifier's
    # precise decision boundary on synthetic data is a fragile thing to
    # pin down. What matters is that it's actually discriminating, not
    # abstaining or picking one direction forever.
    directions = set()
    for cutoff in range(MIN_CANDLES + 10, len(candles), 5):
        signal = strategy.on_candles(candles[:cutoff])
        if signal is not None:
            assert 0 <= signal.strength <= 1
            directions.add(signal.direction.value)
    assert directions == {"long", "short"}


def test_ml_strategy_returns_none_before_warmup(tmp_path) -> None:
    closes = make_cyclical_closes(n_cycles=20, half_len=15)
    candles = make_candles(closes)
    model, _report = train_model(candles, horizon=5, test_fraction=0.2)
    model_path = str(tmp_path / "model.joblib")
    save_model(model, model_path)

    strategy = MLStrategy({"model_path": model_path})
    assert strategy.on_candles(candles[: MIN_CANDLES - 1]) is None


def test_training_report_feature_importances_cover_every_feature() -> None:
    closes = make_cyclical_closes(n_cycles=20, half_len=15)
    candles = make_candles(closes)

    _model, report = train_model(candles, horizon=5, test_fraction=0.2)

    assert {name for name, _ in report.feature_importances} == set(FEATURE_NAMES)
    assert len(report.feature_importances) == len(FEATURE_NAMES)
    total = sum(importance for _, importance in report.feature_importances)
    assert total == pytest.approx(1.0, abs=1e-6)
    # Sorted highest first.
    importances = [importance for _, importance in report.feature_importances]
    assert importances == sorted(importances, reverse=True)


def make_long_cyclical_candles(n_cycles: int = 60, half_len: int = 15, step: float = 1.0) -> list[Candle]:
    return make_candles(make_cyclical_closes(n_cycles=n_cycles, half_len=half_len, step=step))


def test_walk_forward_train_tests_only_on_data_after_its_own_training_window() -> None:
    candles = make_long_cyclical_candles()  # 1800 bars

    folds = walk_forward_train(candles, horizon=5, n_splits=4)

    assert len(folds) == 3  # n_splits - 1
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.training_report.samples > 0


def test_walk_forward_train_compounds_balance_across_folds() -> None:
    candles = make_long_cyclical_candles()

    folds = walk_forward_train(candles, horizon=5, n_splits=4, starting_balance=Decimal(10000))

    assert folds[0].out_of_sample.starting_balance == Decimal(10000)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.out_of_sample.starting_balance == prev.out_of_sample.ending_balance

    summary = walk_forward_train_summary(folds)
    assert summary.starting_balance == folds[0].out_of_sample.starting_balance
    assert summary.ending_balance == folds[-1].out_of_sample.ending_balance
    assert summary.total_trades == sum(f.out_of_sample.total_trades for f in folds)


def test_walk_forward_train_requires_at_least_two_splits() -> None:
    candles = make_long_cyclical_candles()
    with pytest.raises(ValueError):
        walk_forward_train(candles, horizon=5, n_splits=1)


def test_walk_forward_train_summary_of_no_folds_is_empty_result() -> None:
    result = walk_forward_train_summary([])
    assert result.total_trades == 0
    assert result.starting_balance == Decimal(0)


def test_save_and_load_model_round_trip(tmp_path) -> None:
    closes = make_cyclical_closes(n_cycles=20, half_len=15)
    candles = make_candles(closes)
    model, _report = train_model(candles, horizon=5, test_fraction=0.2)
    model_path = str(tmp_path / "roundtrip.joblib")

    save_model(model, model_path)
    loaded = load_model(model_path)

    features = extract_features(candles)
    original_proba = model.predict_proba([features])
    loaded_proba = loaded.predict_proba([features])
    assert list(original_proba[0]) == list(loaded_proba[0])
