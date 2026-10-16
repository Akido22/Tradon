"""Backtesting engine (see ARCHITECTURE.md §2 Этап 6).

Replays a strategy over historical candles using the exact same Strategy
and RiskManager code that runs live — there is no separate "backtest
strategy" implementation, so results reflect what signal-only mode would
actually have produced. Signal-only by design: this simulates PnL, it
never places real orders (see ARCHITECTURE.md §0/§6 on auto-trading risk).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from tradon.data.base import Candle
from tradon.risk.manager import PositionPlan, RiskManager
from tradon.strategy.base import SignalDirection, Strategy
from tradon.strategy.indicators import atr

ATR_PERIOD = 14


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    symbol: str
    direction: SignalDirection
    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    size: Decimal
    pnl: Decimal
    exit_reason: str  # "stop_loss" | "take_profit" | "end_of_data"


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    starting_balance: Decimal = Decimal(0)
    ending_balance: Decimal = Decimal(0)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> Decimal:
        return Decimal(self.wins) / Decimal(self.total_trades) if self.trades else Decimal(0)

    @property
    def profit_factor(self) -> Decimal | None:
        gross_profit = sum((t.pnl for t in self.trades if t.pnl > 0), Decimal(0))
        gross_loss = -sum((t.pnl for t in self.trades if t.pnl < 0), Decimal(0))
        if gross_loss == 0:
            return None  # undefined — no losing trades to divide by
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> Decimal:
        """Largest peak-to-trough decline in equity, as a fraction (0.2 = 20%)."""
        equity = self.starting_balance
        peak = equity
        worst = Decimal(0)
        for t in self.trades:
            equity += t.pnl
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak)
        return worst

    @property
    def total_pnl(self) -> Decimal:
        return self.ending_balance - self.starting_balance

    @property
    def sharpe_ratio(self) -> Decimal | None:
        """Annualized mean per-trade return over its standard deviation —
        the standard risk-adjusted return metric. Uses per-trade returns
        (pnl / equity right before that trade) as the return series and
        annualizes by trade frequency (trades/year, from the actual span
        between the first entry and last exit), since trades close at
        irregular intervals rather than on a fixed calendar schedule. This
        is an approximation — a real Sharpe is usually computed on daily
        returns — but it's consistent and honest about what it measures.
        None when there isn't enough data to mean anything (fewer than 2
        trades, a zero-length time span, or zero variance).
        """
        return self._risk_adjusted_ratio(downside_only=False)

    @property
    def sortino_ratio(self) -> Decimal | None:
        """Same idea as sharpe_ratio, but only penalizes downside
        volatility (losing trades) instead of all variance — a strategy
        with big irregular wins and small steady losses looks better here
        than under Sharpe, which is arguably fairer for a directional
        trading strategy. None if there's no losing trade to measure
        downside from (undefined, not "infinite" — same convention as
        profit_factor for zero gross loss).
        """
        return self._risk_adjusted_ratio(downside_only=True)

    def _risk_adjusted_ratio(self, downside_only: bool) -> Decimal | None:
        if len(self.trades) < 2:
            return None
        equity = self.starting_balance
        returns = []
        for t in self.trades:
            if equity == 0:
                return None
            returns.append(t.pnl / equity)
            equity += t.pnl
        mean = sum(returns) / Decimal(len(returns))

        if downside_only:
            deviations = [r for r in returns if r < 0]  # downside deviation from 0, the usual Sortino convention
        else:
            deviations = [r - mean for r in returns]
        if not deviations:
            return None
        variance = sum(d * d for d in deviations) / Decimal(len(deviations))
        std = variance.sqrt()
        if std == 0:
            return None

        span_days = (self.trades[-1].exit_time - self.trades[0].entry_time).total_seconds() / 86400
        if span_days <= 0:
            return None
        trades_per_year = Decimal(len(self.trades)) / Decimal(str(span_days)) * Decimal(365)
        return (mean / std) * trades_per_year.sqrt()

    @property
    def equity_curve(self) -> list[tuple[datetime, Decimal]]:
        """Cumulative account balance after each trade closes, in order.

        Assumes `self.trades` is already chronological (true for a single
        `run_backtest()` call; `combine_results()` sorts by exit time too).
        """
        equity = self.starting_balance
        curve = []
        for t in self.trades:
            equity += t.pnl
            curve.append((t.exit_time, equity))
        return curve


def run_backtest(
    candles: list[Candle],
    strategy: Strategy,
    risk_manager: RiskManager | None = None,
    starting_balance: Decimal = Decimal(10000),
) -> BacktestResult:
    """Replay `candles` (oldest first) through `strategy`, one position at a
    time. A signal is evaluated on candles up to (not including) index i,
    then fills at candle i's open — no look-ahead. Exits check each bar's
    high/low against the position's stop-loss/take-profit.
    """
    risk_manager = risk_manager or RiskManager()
    result = BacktestResult(starting_balance=starting_balance, ending_balance=starting_balance)

    open_plan: PositionPlan | None = None
    open_entry_time: datetime | None = None

    for i in range(1, len(candles)):
        candle = candles[i]

        if open_plan is not None:
            exit_price, exit_reason = _check_exit(open_plan, candle)
            if exit_price is not None:
                pnl = _pnl(open_plan, exit_price)
                result.trades.append(
                    BacktestTrade(
                        symbol=open_plan.symbol,
                        direction=open_plan.direction,
                        entry_time=open_entry_time,
                        entry_price=open_plan.entry,
                        exit_time=candle.timestamp,
                        exit_price=exit_price,
                        size=open_plan.size,
                        pnl=pnl,
                        exit_reason=exit_reason,
                    )
                )
                result.ending_balance += pnl
                open_plan = None

        if open_plan is None:
            history = candles[:i]
            signal = strategy.on_candles(history)
            if signal is not None:
                current_atr = atr(
                    [c.high for c in history], [c.low for c in history], [c.close for c in history], ATR_PERIOD
                )
                if current_atr and current_atr > 0:
                    open_plan = risk_manager.plan(signal, candle.open, result.ending_balance, current_atr)
                    open_entry_time = candle.timestamp

    if open_plan is not None:
        last = candles[-1]
        pnl = _pnl(open_plan, last.close)
        result.trades.append(
            BacktestTrade(
                symbol=open_plan.symbol,
                direction=open_plan.direction,
                entry_time=open_entry_time,
                entry_price=open_plan.entry,
                exit_time=last.timestamp,
                exit_price=last.close,
                size=open_plan.size,
                pnl=pnl,
                exit_reason="end_of_data",
            )
        )
        result.ending_balance += pnl

    return result


@dataclass(frozen=True, slots=True)
class OptimizationRun:
    params: dict
    result: BacktestResult


def optimize(
    candles: list[Candle],
    strategy_cls: type[Strategy],
    param_grid: dict[str, list[int]],
    risk_manager: RiskManager | None = None,
    starting_balance: Decimal = Decimal(10000),
    valid: Callable[[dict], bool] = lambda params: True,
) -> list[OptimizationRun]:
    """Brute-force grid search: run_backtest() once per combination in
    `param_grid` (e.g. {"fast_period": [5, 10], "slow_period": [20, 30]}).
    `valid` filters out nonsensical combos (e.g. fast >= slow) before
    spending time backtesting them. Results are NOT ranked here — the
    caller decides what "best" means (total PnL, profit factor, trade
    count, ...); see the CLI for the default ranking.
    """
    keys = list(param_grid.keys())
    runs = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo, strict=True))
        if not valid(params):
            continue
        strategy = strategy_cls(params)
        result = run_backtest(candles, strategy, risk_manager, starting_balance)
        runs.append(OptimizationRun(params=params, result=result))
    return runs


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    params: dict
    in_sample: BacktestResult
    out_of_sample: BacktestResult


def walk_forward(
    candles: list[Candle],
    strategy_cls: type[Strategy],
    param_grid: dict[str, list[int]],
    n_splits: int = 5,
    rank_key: Callable[[BacktestResult], Decimal] = lambda r: r.total_pnl,
    risk_manager: RiskManager | None = None,
    starting_balance: Decimal = Decimal(10000),
    valid: Callable[[dict], bool] = lambda params: True,
) -> list[WalkForwardFold]:
    """Anchored (expanding-window) walk-forward validation.

    `optimize()` picks whichever params look best on the very data it
    grades them against — the classic overfitting trap, since a grid
    search will always find *some* combination that happened to fit that
    window's noise. This instead splits `candles` into `n_splits`
    consecutive chunks and, for each fold, optimizes only on data up
    through the chunk before it ("train"), then backtests ONLY that one
    chosen params on the next chunk ("test") — data the optimizer never
    saw. Each fold's test starts from the balance the previous fold's test
    ended on (real compounding), so the aggregate is what re-optimizing
    periodically and only ever trading forward would actually have made —
    a much more honest number than optimize()'s in-sample table.
    """
    if n_splits < 2:
        raise ValueError("walk_forward needs at least 2 splits (1 train, 1 test)")
    boundaries = [round(len(candles) * k / n_splits) for k in range(n_splits + 1)]

    folds: list[WalkForwardFold] = []
    balance = starting_balance
    for i in range(n_splits - 1):
        train = candles[boundaries[0] : boundaries[i + 1]]
        test = candles[boundaries[i + 1] : boundaries[i + 2]]
        if len(train) < 2 or len(test) < 2:
            continue

        runs = optimize(train, strategy_cls, param_grid, risk_manager, balance, valid)
        if not runs:
            continue
        best = max(runs, key=lambda r: rank_key(r.result))

        strategy = strategy_cls(best.params)
        out_of_sample = run_backtest(test, strategy, risk_manager, balance)
        folds.append(
            WalkForwardFold(
                train_start=train[0].timestamp,
                train_end=train[-1].timestamp,
                test_start=test[0].timestamp,
                test_end=test[-1].timestamp,
                params=best.params,
                in_sample=best.result,
                out_of_sample=out_of_sample,
            )
        )
        balance = out_of_sample.ending_balance

    return folds


def walk_forward_summary(folds: list[WalkForwardFold]) -> BacktestResult:
    """Chain every fold's out-of-sample result into one equity curve —
    valid because the test windows are consecutive and non-overlapping by
    construction, so this is real sequential compounding, not the
    independent-runs approximation `combine_results()` makes for concurrent
    instruments.
    """
    if not folds:
        return BacktestResult()
    trades = [t for fold in folds for t in fold.out_of_sample.trades]
    return BacktestResult(
        trades=trades,
        starting_balance=folds[0].out_of_sample.starting_balance,
        ending_balance=folds[-1].out_of_sample.ending_balance,
    )


def run_portfolio(
    candles_by_symbol: dict[str, list[Candle]],
    combos: list[tuple[str, str, Strategy]],
    starting_balance: Decimal = Decimal(10000),
    weights: dict[str, Decimal] | None = None,
    risk_manager: RiskManager | None = None,
) -> tuple[list[tuple[str, str, BacktestResult]], BacktestResult]:
    """Backtest each (symbol, strategy_name, strategy) combo in isolation
    (its own full `starting_balance`, for "how would this one do alone"),
    then separately run them all together as one real portfolio via
    `simulate_portfolio()` for `combined` — see that function for why this
    isn't just `combine_results()` on the isolated runs. Returns (per-combo
    isolated results, combined shared-capital result).
    """
    runs: list[tuple[str, str, BacktestResult]] = []
    valid_combos: list[tuple[str, str, Strategy]] = []
    for symbol, strategy_name, strategy in combos:
        candles = candles_by_symbol.get(symbol)
        if not candles:
            continue
        valid_combos.append((symbol, strategy_name, strategy))
        result = run_backtest(candles, strategy, starting_balance=starting_balance)
        runs.append((symbol, strategy_name, result))

    combined = (
        simulate_portfolio(candles_by_symbol, valid_combos, starting_balance, weights, risk_manager)
        if valid_combos
        else BacktestResult()
    )
    return runs, combined


def combine_results(results: list[BacktestResult]) -> BacktestResult:
    """Merge several backtests (e.g. one per symbol/strategy) into a single
    portfolio-level result — each combo trades with its own full starting
    balance rather than a shared allocation, so this approximates running
    them side by side, not true capital-split portfolio simulation. Prefer
    `simulate_portfolio()` (what `run_portfolio()`'s `combined` return uses)
    when you actually want a shared account; this stays around as a plain
    "just add the numbers up" utility.
    """
    trades = sorted((t for r in results for t in r.trades), key=lambda t: t.exit_time)
    return BacktestResult(
        trades=trades,
        starting_balance=sum((r.starting_balance for r in results), Decimal(0)),
        ending_balance=sum((r.ending_balance for r in results), Decimal(0)),
    )


def combo_key(symbol: str, strategy_name: str) -> str:
    return f"{symbol}/{strategy_name}"


def correlation_matrix(candles_by_symbol: dict[str, list[Candle]]) -> dict[str, dict[str, Decimal]]:
    """Pearson correlation of close-to-close returns between every pair of
    symbols — how similarly two instruments MOVE, independent of whatever
    strategy you run on them. Two instruments correlated near +1 carry
    largely the same risk twice over, so running the same directional
    strategy on both isn't real diversification even though it looks like
    two positions; near -1 means they tend to move opposite each other.
    Candles are matched up by timestamp (only bars every symbol actually
    has get compared), so mismatched history lengths or gaps are fine.

    Returns a symbol -> symbol -> correlation dict (both directions
    filled in, diagonal is always 1). A pair is simply absent from the
    result if they don't have at least 2 overlapping bars or either side
    has zero variance (e.g. a flat price feed) — there's nothing
    meaningful to compute there.
    """
    returns_by_symbol: dict[str, dict[datetime, Decimal]] = {}
    for symbol, candles in candles_by_symbol.items():
        returns: dict[datetime, Decimal] = {}
        for prev, cur in zip(candles, candles[1:], strict=False):
            if prev.close != 0:
                returns[cur.timestamp] = (cur.close - prev.close) / prev.close
        returns_by_symbol[symbol] = returns

    symbols = list(candles_by_symbol)
    matrix: dict[str, dict[str, Decimal]] = {symbol: {symbol: Decimal(1)} for symbol in symbols}
    for a, b in itertools.combinations(symbols, 2):
        common_times = returns_by_symbol[a].keys() & returns_by_symbol[b].keys()
        if len(common_times) < 2:
            continue
        xs = [returns_by_symbol[a][t] for t in common_times]
        ys = [returns_by_symbol[b][t] for t in common_times]
        corr = _pearson(xs, ys)
        if corr is not None:
            matrix[a][b] = corr
            matrix[b][a] = corr
    return matrix


def _pearson(xs: list[Decimal], ys: list[Decimal]) -> Decimal | None:
    n = Decimal(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / n
    variance_x = sum((x - mean_x) ** 2 for x in xs) / n
    variance_y = sum((y - mean_y) ** 2 for y in ys) / n
    if variance_x == 0 or variance_y == 0:
        return None
    return covariance / (variance_x * variance_y).sqrt()


def simulate_portfolio(
    candles_by_symbol: dict[str, list[Candle]],
    combos: list[tuple[str, str, Strategy]],
    starting_balance: Decimal = Decimal(10000),
    weights: dict[str, Decimal] | None = None,
    risk_manager: RiskManager | None = None,
) -> BacktestResult:
    """Real shared-capital portfolio simulation: every combo trades against
    ONE account balance instead of each pretending it has the full starting
    balance to itself (that's what `combine_results()` does, and it isn't
    realistic — a drawdown in one instrument doesn't actually shrink what
    another instrument can risk next under that approach).

    All combos' candles are walked on a single merged chronological
    timeline (interleaved by timestamp, not run one after another), so
    positions on different instruments can be open at the same time,
    competing for the same capital exactly as a real multi-instrument
    account would. Each combo's risk-per-trade is sized off
    `weights[combo_key] * current shared balance` at the moment its signal
    fires — `weights` defaults to an equal split across combos.
    """
    risk_manager = risk_manager or RiskManager()
    keys = [combo_key(symbol, strategy.name) for symbol, _, strategy in combos]
    if weights is None:
        weights = {key: Decimal(1) / Decimal(len(combos)) for key in keys} if combos else {}

    result = BacktestResult(starting_balance=starting_balance, ending_balance=starting_balance)
    if not combos:
        return result

    states = [
        {
            "key": key,
            "candles": candles_by_symbol.get(symbol, []),
            "strategy": strategy,
            "open_plan": None,
            "open_entry_time": None,
        }
        for (symbol, _, strategy), key in zip(combos, keys, strict=True)
    ]

    # One global, chronologically-sorted timeline of (timestamp, combo
    # index, bar index) so exits/entries across different instruments
    # interleave in real time order instead of each combo replaying
    # start-to-finish on its own before the next one begins.
    events = sorted(
        (state["candles"][i].timestamp, si, i)
        for si, state in enumerate(states)
        for i in range(1, len(state["candles"]))
    )

    balance = starting_balance
    for _, si, i in events:
        state = states[si]
        candle = state["candles"][i]
        plan = state["open_plan"]

        if plan is not None:
            exit_price, exit_reason = _check_exit(plan, candle)
            if exit_price is not None:
                pnl = _pnl(plan, exit_price)
                result.trades.append(
                    BacktestTrade(
                        symbol=plan.symbol,
                        direction=plan.direction,
                        entry_time=state["open_entry_time"],
                        entry_price=plan.entry,
                        exit_time=candle.timestamp,
                        exit_price=exit_price,
                        size=plan.size,
                        pnl=pnl,
                        exit_reason=exit_reason,
                    )
                )
                balance += pnl
                state["open_plan"] = None
                plan = None

        if plan is None:
            history = state["candles"][:i]
            signal = state["strategy"].on_candles(history)
            if signal is not None:
                current_atr = atr(
                    [c.high for c in history], [c.low for c in history], [c.close for c in history], ATR_PERIOD
                )
                if current_atr and current_atr > 0:
                    allocated_balance = balance * weights.get(state["key"], Decimal(0))
                    new_plan = risk_manager.plan(signal, candle.open, allocated_balance, current_atr)
                    if new_plan.size > 0:
                        state["open_plan"] = new_plan
                        state["open_entry_time"] = candle.timestamp

    # Anything still open when its own candles run out closes at last close.
    for state in states:
        plan = state["open_plan"]
        if plan is not None and state["candles"]:
            last = state["candles"][-1]
            pnl = _pnl(plan, last.close)
            result.trades.append(
                BacktestTrade(
                    symbol=plan.symbol,
                    direction=plan.direction,
                    entry_time=state["open_entry_time"],
                    entry_price=plan.entry,
                    exit_time=last.timestamp,
                    exit_price=last.close,
                    size=plan.size,
                    pnl=pnl,
                    exit_reason="end_of_data",
                )
            )
            balance += pnl

    result.trades.sort(key=lambda t: t.exit_time)
    result.ending_balance = balance
    return result


def _check_exit(plan: PositionPlan, candle: Candle) -> tuple[Decimal | None, str]:
    if plan.direction is SignalDirection.LONG:
        hit_stop = candle.low <= plan.stop_loss
        hit_target = candle.high >= plan.take_profit
    else:
        hit_stop = candle.high >= plan.stop_loss
        hit_target = candle.low <= plan.take_profit

    if hit_stop:
        # Conservative: if both were touched in the same bar, assume the
        # worse outcome rather than crediting a lucky take-profit.
        return plan.stop_loss, "stop_loss"
    if hit_target:
        return plan.take_profit, "take_profit"
    return None, ""


def _pnl(plan: PositionPlan, exit_price: Decimal) -> Decimal:
    if plan.direction is SignalDirection.LONG:
        return (exit_price - plan.entry) * plan.size
    return (plan.entry - exit_price) * plan.size
