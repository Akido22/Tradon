from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tradon.backtest import (
    BacktestResult,
    BacktestTrade,
    combine_results,
    combo_key,
    correlation_matrix,
    optimize,
    run_backtest,
    run_portfolio,
    simulate_portfolio,
    walk_forward,
    walk_forward_summary,
)
from tradon.data.base import Candle
from tradon.strategy.base import SignalDirection
from tradon.strategy.plugins.sma_cross import SmaCrossStrategy


def make_trending_candles(closes: list[float], spread: float = 0.05) -> list[Candle]:
    """Candles with real high/low range (unlike conftest's flat fixture) so ATR > 0."""
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


def test_backtest_uptrend_produces_profitable_long() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)

    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})
    result = run_backtest(candles, strategy, starting_balance=Decimal(10000))

    assert result.total_trades > 0
    assert any(t.direction is SignalDirection.LONG for t in result.trades)
    assert result.ending_balance != result.starting_balance
    assert 0 <= result.win_rate <= 1
    assert result.max_drawdown >= 0


def test_backtest_no_signals_means_no_trades() -> None:
    flat_candles = make_trending_candles([100.0] * 20)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    result = run_backtest(flat_candles, strategy)

    assert result.total_trades == 0
    assert result.ending_balance == result.starting_balance
    assert result.profit_factor is None
    assert result.win_rate == 0


def test_optimize_runs_every_valid_grid_combo() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    grid = {"fast_period": [5, 8], "slow_period": [10, 20]}

    runs = optimize(
        candles, SmaCrossStrategy, grid, valid=lambda p: p["fast_period"] < p["slow_period"]
    )

    assert len(runs) == 4  # all 4 combos have fast < slow
    assert all(r.params["fast_period"] < r.params["slow_period"] for r in runs)
    assert all(isinstance(r.result.total_trades, int) for r in runs)


def test_optimize_skips_invalid_combos() -> None:
    candles = make_trending_candles([100.0] * 20)
    grid = {"fast_period": [10, 20], "slow_period": [10, 20]}

    runs = optimize(candles, SmaCrossStrategy, grid, valid=lambda p: p["fast_period"] < p["slow_period"])

    assert len(runs) == 1
    assert runs[0].params == {"fast_period": 10, "slow_period": 20}


def test_combine_results_sums_balances_and_sorts_trades_by_exit_time() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    result_a = run_backtest(candles, strategy, starting_balance=Decimal(1000))
    result_b = run_backtest(candles, strategy, starting_balance=Decimal(2000))

    combined = combine_results([result_a, result_b])

    assert combined.starting_balance == Decimal(3000)
    assert combined.ending_balance == result_a.ending_balance + result_b.ending_balance
    assert combined.total_trades == result_a.total_trades + result_b.total_trades
    exit_times = [t.exit_time for t in combined.trades]
    assert exit_times == sorted(exit_times)


def test_equity_curve_reflects_cumulative_pnl() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    result = run_backtest(candles, strategy, starting_balance=Decimal(10000))
    curve = result.equity_curve

    assert len(curve) == result.total_trades
    assert curve[-1][1] == result.ending_balance
    running = result.starting_balance
    for (_, equity), trade in zip(curve, result.trades, strict=True):
        running += trade.pnl
        assert equity == running


def test_simulate_portfolio_single_combo_matches_isolated_backtest() -> None:
    """With one combo, its equal-split weight is 1.0 — the shared-capital
    simulation should size and trade identically to a plain run_backtest."""
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    isolated = run_backtest(candles, strategy, starting_balance=Decimal(10000))
    combined = simulate_portfolio({"TEST": candles}, [("TEST", "sma_cross", strategy)], starting_balance=Decimal(10000))

    assert combined.total_trades == isolated.total_trades
    assert combined.ending_balance == isolated.ending_balance
    assert [t.pnl for t in combined.trades] == [t.pnl for t in isolated.trades]


def test_simulate_portfolio_shares_capital_across_combos() -> None:
    """Two combos on the same shared account should risk less per trade
    than either would alone with the full balance (equal split halves
    the capital each one sizes its risk against)."""
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    strategy_a = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})
    strategy_b = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    isolated = run_backtest(candles, strategy_a, starting_balance=Decimal(10000))
    combined = simulate_portfolio(
        {"A": candles, "B": candles},
        [("A", "sma_cross", strategy_a), ("B", "sma_cross", strategy_b)],
        starting_balance=Decimal(10000),
    )

    assert isolated.total_trades > 0
    assert combined.total_trades == isolated.total_trades * 2
    first_combined_size = combined.trades[0].size
    first_isolated_size = isolated.trades[0].size
    assert first_combined_size < first_isolated_size  # half the capital -> smaller position


def test_simulate_portfolio_zero_weight_combo_never_trades() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    strategy_a = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})
    strategy_b = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    isolated = run_backtest(candles, strategy_a, starting_balance=Decimal(10000))
    combined = simulate_portfolio(
        {"A": candles, "B": candles},
        [("A", "sma_cross", strategy_a), ("B", "sma_cross", strategy_b)],
        starting_balance=Decimal(10000),
        weights={combo_key("A", "sma_cross"): Decimal(1), combo_key("B", "sma_cross"): Decimal(0)},
    )

    # Only the fully-weighted combo (A) should trade — B's zero weight means
    # every risk_manager.plan() for it sizes to 0, so it never opens.
    assert combined.total_trades == isolated.total_trades
    assert combined.total_trades > 0


def test_sharpe_and_sortino_are_none_below_two_trades() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 12)]  # short — likely 0 or 1 trade
    candles = make_trending_candles(flat + uptrend)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    result = run_backtest(candles, strategy, starting_balance=Decimal(10000))
    if result.total_trades < 2:
        assert result.sharpe_ratio is None
        assert result.sortino_ratio is None


def test_sharpe_and_sortino_are_defined_for_a_normal_run() -> None:
    import math

    closes = [100.0 + 5 * math.sin(i / 6) + 0.05 * i for i in range(200)]
    candles = make_trending_candles(closes)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    result = run_backtest(candles, strategy, starting_balance=Decimal(10000))

    assert result.total_trades >= 2
    assert result.sharpe_ratio is not None
    # Sortino needs at least one losing trade to have a downside sample —
    # only assert it when this particular run actually produced one.
    if result.losses > 0:
        assert result.sortino_ratio is not None


def _make_trade(entry_time: datetime, pnl: Decimal) -> BacktestTrade:
    return BacktestTrade(
        symbol="TEST",
        direction=SignalDirection.LONG,
        entry_time=entry_time,
        entry_price=Decimal(100),
        exit_time=entry_time + timedelta(hours=1),
        exit_price=Decimal(100) + pnl,
        size=Decimal(1),
        pnl=pnl,
        exit_reason="take_profit",
    )


def test_sortino_is_none_with_no_losing_trades() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [_make_trade(start + timedelta(hours=i * 3), Decimal(50 + i)) for i in range(5)]
    result = BacktestResult(trades=trades, starting_balance=Decimal(10000), ending_balance=Decimal(10000) + sum(t.pnl for t in trades))

    assert result.losses == 0
    assert result.sortino_ratio is None
    assert result.sharpe_ratio is not None  # Sharpe still works — it uses all variance, not just downside


def test_correlation_matrix_identical_series_is_one() -> None:
    candles = make_trending_candles([100.0 + i * 0.3 for i in range(50)])
    matrix = correlation_matrix({"A": candles, "B": candles})

    assert matrix["A"]["A"] == Decimal(1)
    assert matrix["A"]["B"] == Decimal(1)
    assert matrix["B"]["A"] == Decimal(1)


def test_correlation_matrix_inverse_series_is_minus_one() -> None:
    # Build B's returns as the exact negative of A's returns (not just "a
    # price line sloping the other way" — a linear down-trend's own pct
    # returns don't work out to literally -1x a linear up-trend's, since
    # both shrink toward 0 as the base grows, in the SAME direction).
    a_closes = [100.0]
    for i in range(49):
        a_closes.append(a_closes[-1] * (1 + 0.01 * ((i % 5) - 2)))
    b_closes = [100.0]
    for i in range(1, len(a_closes)):
        r = (a_closes[i] - a_closes[i - 1]) / a_closes[i - 1]
        b_closes.append(b_closes[-1] * (1 - r))

    matrix = correlation_matrix({"A": make_trending_candles(a_closes), "B": make_trending_candles(b_closes)})

    assert matrix["A"]["B"] < Decimal("-0.99")


def test_correlation_matrix_skips_pairs_without_enough_overlap() -> None:
    a = make_trending_candles([100.0 + i * 0.3 for i in range(50)])
    b = [a[0]]  # only one candle — no returns computable at all
    matrix = correlation_matrix({"A": a, "B": b})

    assert "B" not in matrix["A"]


def make_oscillating_candles(n: int, spread: float = 0.05) -> list[Candle]:
    """A wandering price path (not a single clean trend) so SMA crossovers —
    and therefore trades — show up throughout the whole series, including
    inside every walk-forward fold, not just near the end."""
    import math

    closes = [100.0 + 3 * math.sin(i / 6) + 0.05 * i for i in range(n)]
    return make_trending_candles(closes, spread)


def test_walk_forward_tests_only_on_data_after_its_own_training_window() -> None:
    candles = make_oscillating_candles(200)
    grid = {"fast_period": [3, 5], "slow_period": [10, 15]}
    valid = lambda p: p["fast_period"] < p["slow_period"]  # noqa: E731

    folds = walk_forward(candles, SmaCrossStrategy, grid, n_splits=4, valid=valid)

    assert len(folds) == 3  # n_splits - 1
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.params["fast_period"] < fold.params["slow_period"]
        assert fold.params["fast_period"] in grid["fast_period"]


def test_walk_forward_compounds_balance_across_folds() -> None:
    candles = make_oscillating_candles(200)
    grid = {"fast_period": [3, 5], "slow_period": [10, 15]}
    valid = lambda p: p["fast_period"] < p["slow_period"]  # noqa: E731

    folds = walk_forward(candles, SmaCrossStrategy, grid, n_splits=4, starting_balance=Decimal(10000), valid=valid)

    assert folds[0].out_of_sample.starting_balance == Decimal(10000)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.out_of_sample.starting_balance == prev.out_of_sample.ending_balance

    summary = walk_forward_summary(folds)
    assert summary.starting_balance == folds[0].out_of_sample.starting_balance
    assert summary.ending_balance == folds[-1].out_of_sample.ending_balance
    assert summary.total_trades == sum(f.out_of_sample.total_trades for f in folds)


def test_walk_forward_requires_at_least_two_splits() -> None:
    candles = make_oscillating_candles(50)
    try:
        walk_forward(candles, SmaCrossStrategy, {"fast_period": [5], "slow_period": [10]}, n_splits=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_walk_forward_summary_of_no_folds_is_empty_result() -> None:
    result = walk_forward_summary([])
    assert result.total_trades == 0
    assert result.starting_balance == Decimal(0)


def test_run_portfolio_skips_symbols_without_candles() -> None:
    flat = [100.0] * 15
    uptrend = [100.0 + i * 0.5 for i in range(1, 40)]
    candles = make_trending_candles(flat + uptrend)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    runs, combined = run_portfolio(
        candles_by_symbol={"HAS_DATA": candles, "NO_DATA": []},
        combos=[
            ("HAS_DATA", "sma_cross", strategy),
            ("NO_DATA", "sma_cross", strategy),
        ],
        starting_balance=Decimal(5000),
    )

    assert len(runs) == 1
    assert runs[0][0] == "HAS_DATA"
    assert combined.starting_balance == Decimal(5000)
    assert combined.total_trades == runs[0][2].total_trades
