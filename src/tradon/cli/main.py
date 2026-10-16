"""CLI — Presentation Layer (see ARCHITECTURE.md §3.7).

Contains no analysis logic: it only calls tradon.runtime (the core) and
renders the result with Rich. GUI does the same against the same core,
without touching this module.
"""
from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable
from decimal import Decimal

import typer
from rich.console import Console
from rich.table import Table

from tradon.backtest import combo_key, correlation_matrix, optimize, run_portfolio, walk_forward, walk_forward_summary
from tradon.runtime import (
    PROVIDERS,
    STRATEGIES,
    LiveSession,
    build_provider,
    build_risk_manager,
    build_signal_engine,
    build_strategies,
    describe_error,
    load_runtime_config,
    normalize_symbol,
    poll_once,
)
from tradon.db.session import get_engine, get_sessionmaker

app = typer.Typer(help="Tradon — signal-only trading bot core (CLI presentation layer)")
console = Console()


@app.command()
def symbols(
    provider: str = typer.Option("bybit", "--provider", help=f"Data source: {', '.join(PROVIDERS)}"),
) -> None:
    """List symbols available from a data provider."""
    asyncio.run(_symbols(provider))


async def _symbols(provider_name: str) -> None:
    if provider_name not in PROVIDERS:
        raise typer.BadParameter(f"Unknown provider {provider_name!r} — choices: {', '.join(PROVIDERS)}")
    provider = PROVIDERS[provider_name]()
    try:
        syms = await provider.get_symbols()
    except NotImplementedError as exc:
        raise typer.BadParameter(f"{provider.name} isn't wired up yet: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean message, not a raw traceback
        raise typer.BadParameter(f"Could not reach {provider.name}: {describe_error(exc)}") from exc
    for symbol in syms:
        console.print(symbol)


@app.command()
def run(
    provider: str = typer.Option(
        None, "--provider", help=f"Data source ({', '.join(PROVIDERS)}) — overrides config.yaml provider"
    ),
    symbols: str = typer.Option(
        None, "--symbols", help="Comma-separated symbols (e.g. EURUSD,GBPUSD) — overrides config.yaml instruments"
    ),
    timeframe: str = typer.Option(
        None, "--timeframe", help="Overrides config.yaml timeframe — e.g. 1m while testing, so bars close fast"
    ),
    once: bool = typer.Option(False, "--once", help="Run a single poll cycle then exit instead of monitoring"),
) -> None:
    """Continuously poll instruments, evaluate strategies, print + persist signals (Ctrl+C to stop)."""
    asyncio.run(_run(once, symbols, provider, timeframe))


async def _run(once: bool, symbols_csv: str | None, provider_name: str | None, timeframe: str | None) -> None:
    cfg = load_runtime_config()
    if provider_name:
        if provider_name not in PROVIDERS:
            raise typer.BadParameter(f"Unknown provider {provider_name!r} — choices: {', '.join(PROVIDERS)}")
        cfg.provider_name = provider_name
    if timeframe:
        cfg.timeframe = timeframe
    provider = build_provider(cfg)

    if symbols_csv:
        cfg.instruments = [
            normalize_symbol(cfg.provider_name, s) for s in symbols_csv.split(",") if s.strip()
        ]
    if not cfg.instruments:
        raise typer.BadParameter("No symbols given — pass --symbols or set instruments in config.yaml")

    try:
        unknown = [s for s in cfg.instruments if not await provider.supports_symbol(s)]
    except NotImplementedError as exc:
        raise typer.BadParameter(f"{provider.name} isn't wired up yet: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean message, not a raw traceback
        raise typer.BadParameter(f"Could not reach {provider.name}: {describe_error(exc)}") from exc
    if unknown:
        raise typer.BadParameter(f"{provider.name} does not support: {', '.join(unknown)}")

    strategies = build_strategies(cfg)
    signal_engine = build_signal_engine(cfg)
    risk_manager = build_risk_manager(cfg)

    if once:
        signals = await poll_once(
            provider, cfg.instruments, cfg.timeframe, strategies, signal_engine,
            risk_manager, cfg.risk_account_balance,
        )
        _render(signals)
        await get_engine().dispose()
        return

    session = LiveSession(
        provider, cfg.instruments, cfg.timeframe, strategies, signal_engine,
        risk_manager, cfg.risk_account_balance,
        on_signals=_render,
        on_error=lambda exc: console.print(f"[red]Poll failed: {exc}[/red]"),
    )
    session.start(cfg.poll_interval_seconds)
    console.print("[dim]Connecting (realtime if the provider supports it, else polling)...[/dim]")
    console.print(
        f"[dim]A realtime provider only reports a bar once it actually closes — on {cfg.timeframe!r} that can mean "
        "up to that long with no output below. This is expected, not a hang; use --once for an immediate check, "
        "or --timeframe 1m while testing.[/dim]"
    )
    try:
        last_reported = session.mode
        while True:
            await asyncio.sleep(1)
            if session.mode != last_reported:
                last_reported = session.mode
                if session.mode == "realtime":
                    console.print("[dim]Live via the provider's realtime stream (Ctrl+C to stop)...[/dim]")
                elif session.mode == "poll":
                    console.print(
                        f"[dim]Realtime unavailable — polling every {cfg.poll_interval_seconds}s "
                        "(Ctrl+C to stop)...[/dim]"
                    )
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("[yellow]Stopping...[/yellow]")
    finally:
        session.stop()
        await session.wait_closed()
        await get_engine().dispose()


@app.command()
def backtest(
    symbol: str = typer.Option(..., "--symbol", help="Comma-separated symbols, e.g. EURUSD,GBPUSD"),
    strategy_name: str = typer.Option(
        ..., "--strategy", help=f"Comma-separated strategies: {', '.join(STRATEGIES)}"
    ),
    provider: str = typer.Option(None, "--provider", help="Data source — overrides config.yaml provider"),
    timeframe: str = typer.Option(None, "--timeframe", help="Overrides config.yaml timeframe"),
    limit: int = typer.Option(500, "--limit", help="How many historical candles to fetch"),
    balance: float = typer.Option(
        10000, "--balance", help="Shared portfolio starting balance (each combo alone still gets its own full amount)"
    ),
    weights: str = typer.Option(
        None,
        "--weights",
        help=(
            "Comma-separated capital shares for the combined portfolio run, same order as "
            "symbol x strategy (e.g. EURUSD/sma_cross,EURUSD/rsi,GBPUSD/sma_cross,GBPUSD/rsi for "
            "--symbol EURUSD,GBPUSD --strategy sma_cross,rsi). Must sum to 1. Default: equal split."
        ),
    ),
) -> None:
    """Replay strategies over historical candles — no live data, no orders (see ARCHITECTURE.md §6).

    Pass multiple --symbol/--strategy values (comma-separated) to backtest a
    portfolio: each combo also runs in isolation (its own full balance, to
    show standalone quality), then all combos run together sharing one
    account — see simulate_portfolio() in backtest.py for what that means.
    """
    symbols_ = [s.strip() for s in symbol.split(",") if s.strip()]
    strategy_names = [s.strip() for s in strategy_name.split(",") if s.strip()]
    weights_ = [Decimal(w.strip()) for w in weights.split(",") if w.strip()] if weights else None
    asyncio.run(_backtest(symbols_, strategy_names, provider, timeframe, limit, balance, weights_))


async def _backtest(
    symbols: list[str],
    strategy_names: list[str],
    provider_name: str | None,
    timeframe: str | None,
    limit: int,
    balance: float,
    weights: list[Decimal] | None,
) -> None:
    unknown_strategies = [s for s in strategy_names if s not in STRATEGIES]
    if unknown_strategies:
        raise typer.BadParameter(f"Unknown strategy: {', '.join(unknown_strategies)} — choices: {', '.join(STRATEGIES)}")

    cfg = load_runtime_config()
    if provider_name:
        if provider_name not in PROVIDERS:
            raise typer.BadParameter(f"Unknown provider {provider_name!r} — choices: {', '.join(PROVIDERS)}")
        cfg.provider_name = provider_name
    provider = build_provider(cfg)
    timeframe = timeframe or cfg.timeframe
    params_by_strategy = {s["name"]: s.get("params", {}) for s in cfg.strategy_specs}
    symbols = [normalize_symbol(cfg.provider_name, s) for s in symbols]

    candles_by_symbol = {}
    for sym in symbols:
        console.print(f"[dim]Fetching {limit} {timeframe} candles for {sym} from {provider.name}...[/dim]")
        try:
            candles_by_symbol[sym] = await provider.get_ohlcv(sym, timeframe, limit=limit)
        except NotImplementedError as exc:
            raise typer.BadParameter(f"{provider.name} isn't wired up yet: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean message, not a raw traceback
            raise typer.BadParameter(f"Could not fetch {sym} from {provider.name}: {describe_error(exc)}") from exc

    combos = [
        (sym, strategy_name, STRATEGIES[strategy_name](params_by_strategy.get(strategy_name, {})))
        for sym, strategy_name in itertools.product(symbols, strategy_names)
    ]
    weights_by_key = None
    if weights is not None:
        if len(weights) != len(combos):
            raise typer.BadParameter(f"--weights has {len(weights)} value(s) but there are {len(combos)} combo(s)")
        if sum(weights) != 1:
            raise typer.BadParameter(f"--weights must sum to 1 (got {sum(weights)})")
        weights_by_key = {combo_key(sym, name): w for (sym, name, _), w in zip(combos, weights, strict=True)}
    runs, combined = run_portfolio(
        candles_by_symbol, combos, starting_balance=Decimal(str(balance)), weights=weights_by_key
    )

    if not runs:
        console.print("[red]Nothing to backtest.[/red]")
        return

    if len(runs) == 1:
        sym, strategy_name, result = runs[0]
        _render_backtest_metrics(f"Backtest — {sym} / {strategy_name}", result)
        return

    table = Table(title="Backtest — per combo")
    for column in ("Symbol", "Strategy", "Trades", "Win rate", "Profit factor", "PnL"):
        table.add_column(column)
    for sym, strategy_name, result in runs:
        table.add_row(
            sym,
            strategy_name,
            str(result.total_trades),
            f"{result.win_rate:.1%}" if result.trades else "n/a",
            f"{result.profit_factor:.2f}" if result.profit_factor is not None else "n/a",
            f"{result.total_pnl:+.2f}",
        )
    console.print(table)
    _render_correlation_matrix(candles_by_symbol)
    _render_backtest_metrics("Portfolio (shared capital)", combined)


@app.command()
def optimize_params(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to optimize against, e.g. EURUSD"),
    strategy_name: str = typer.Option(..., "--strategy", help=f"Strategy: {', '.join(STRATEGIES)}"),
    provider: str = typer.Option(None, "--provider", help="Data source — overrides config.yaml provider"),
    timeframe: str = typer.Option(None, "--timeframe", help="Overrides config.yaml timeframe"),
    limit: int = typer.Option(1000, "--limit", help="How many historical candles to fetch"),
    balance: float = typer.Option(10000, "--balance", help="Starting account balance"),
    top: int = typer.Option(10, "--top", help="How many best-performing combos to show"),
) -> None:
    """Grid-search a strategy's parameters against history — same no-orders rule as backtest."""
    asyncio.run(_optimize(symbol, strategy_name, provider, timeframe, limit, balance, top))


_OPTIMIZE_GRIDS: dict[str, dict[str, list[int]]] = {
    "sma_cross": {"fast_period": [5, 8, 10, 15, 20], "slow_period": [20, 30, 50, 100]},
    "rsi": {"period": [7, 14, 21], "oversold": [20, 25, 30], "overbought": [70, 75, 80]},
    "macd": {"fast_period": [8, 12, 16], "slow_period": [21, 26, 35], "signal_period": [5, 9, 12]},
    "bollinger": {"period": [10, 20, 30]},
    "atr_breakout": {"channel_period": [10, 20, 30], "atr_period": [10, 14, 21]},
    "ichimoku": {"tenkan_period": [7, 9, 10], "kijun_period": [22, 26, 30]},
}
_OPTIMIZE_VALID: dict[str, Callable[[dict], bool]] = {
    "sma_cross": lambda p: p["fast_period"] < p["slow_period"],
    "macd": lambda p: p["fast_period"] < p["slow_period"],
    "ichimoku": lambda p: p["tenkan_period"] < p["kijun_period"],
}


async def _optimize(
    symbol: str,
    strategy_name: str,
    provider_name: str | None,
    timeframe: str | None,
    limit: int,
    balance: float,
    top: int,
) -> None:
    if strategy_name not in _OPTIMIZE_GRIDS:
        raise typer.BadParameter(f"No default grid for {strategy_name!r} — choices: {', '.join(_OPTIMIZE_GRIDS)}")

    cfg = load_runtime_config()
    if provider_name:
        if provider_name not in PROVIDERS:
            raise typer.BadParameter(f"Unknown provider {provider_name!r} — choices: {', '.join(PROVIDERS)}")
        cfg.provider_name = provider_name
    provider = build_provider(cfg)
    timeframe = timeframe or cfg.timeframe
    symbol = normalize_symbol(cfg.provider_name, symbol)

    console.print(f"[dim]Fetching {limit} {timeframe} candles for {symbol} from {provider.name}...[/dim]")
    try:
        candles = await provider.get_ohlcv(symbol, timeframe, limit=limit)
    except NotImplementedError as exc:
        raise typer.BadParameter(f"{provider.name} isn't wired up yet: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean message, not a raw traceback
        raise typer.BadParameter(f"Could not fetch {symbol} from {provider.name}: {describe_error(exc)}") from exc
    if not candles:
        console.print("[red]No candles returned — nothing to optimize.[/red]")
        return

    grid = _OPTIMIZE_GRIDS[strategy_name]
    valid = _OPTIMIZE_VALID.get(strategy_name, lambda p: True)
    console.print(f"[dim]Trying {len(list(itertools.product(*grid.values())))} combos...[/dim]")
    runs = optimize(candles, STRATEGIES[strategy_name], grid, starting_balance=Decimal(str(balance)), valid=valid)
    runs.sort(key=lambda r: r.result.total_pnl, reverse=True)

    table = Table(title=f"Optimize — {symbol} / {strategy_name} (ranked by total PnL)")
    table.add_column("Params")
    table.add_column("Trades")
    table.add_column("Win rate")
    table.add_column("Profit factor")
    table.add_column("PnL")
    for run in runs[:top]:
        table.add_row(
            ", ".join(f"{k}={v}" for k, v in run.params.items()),
            str(run.result.total_trades),
            f"{run.result.win_rate:.1%}" if run.result.trades else "n/a",
            f"{run.result.profit_factor:.2f}" if run.result.profit_factor is not None else "n/a",
            f"{run.result.total_pnl:+.2f}",
        )
    console.print(table)
    console.print("[dim]Low trade counts are noise, not edge — check they're statistically meaningful.[/dim]")


@app.command("walk-forward")
def walk_forward_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to validate, e.g. EURUSD"),
    strategy_name: str = typer.Option(..., "--strategy", help=f"Strategy: {', '.join(STRATEGIES)}"),
    provider: str = typer.Option(None, "--provider", help="Data source — overrides config.yaml provider"),
    timeframe: str = typer.Option(None, "--timeframe", help="Overrides config.yaml timeframe"),
    limit: int = typer.Option(2000, "--limit", help="How many historical candles to fetch — needs more than backtest/optimize since it's split into folds"),
    balance: float = typer.Option(10000, "--balance", help="Starting account balance"),
    splits: int = typer.Option(5, "--splits", help="Consecutive windows (>=2); each fold optimizes on everything before it, tests only on itself"),
) -> None:
    """Walk-forward validation: optimize-params picks whichever params fit
    a window's own noise best, which overstates how they'd do on data it
    hasn't seen. This re-optimizes per fold on data before it, tests only
    on the fold after (out-of-sample), and chains the folds into one real
    equity curve — same no-orders rule as backtest.
    """
    asyncio.run(_walk_forward(symbol, strategy_name, provider, timeframe, limit, balance, splits))


async def _walk_forward(
    symbol: str,
    strategy_name: str,
    provider_name: str | None,
    timeframe: str | None,
    limit: int,
    balance: float,
    splits: int,
) -> None:
    if strategy_name not in _OPTIMIZE_GRIDS:
        raise typer.BadParameter(f"No default grid for {strategy_name!r} — choices: {', '.join(_OPTIMIZE_GRIDS)}")
    if splits < 2:
        raise typer.BadParameter("--splits must be at least 2 (1 train, 1 test)")

    cfg = load_runtime_config()
    if provider_name:
        if provider_name not in PROVIDERS:
            raise typer.BadParameter(f"Unknown provider {provider_name!r} — choices: {', '.join(PROVIDERS)}")
        cfg.provider_name = provider_name
    provider = build_provider(cfg)
    timeframe = timeframe or cfg.timeframe
    symbol = normalize_symbol(cfg.provider_name, symbol)

    console.print(f"[dim]Fetching {limit} {timeframe} candles for {symbol} from {provider.name}...[/dim]")
    try:
        candles = await provider.get_ohlcv(symbol, timeframe, limit=limit)
    except NotImplementedError as exc:
        raise typer.BadParameter(f"{provider.name} isn't wired up yet: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean message, not a raw traceback
        raise typer.BadParameter(f"Could not fetch {symbol} from {provider.name}: {describe_error(exc)}") from exc
    if len(candles) < splits * 2:
        console.print(f"[red]Only {len(candles)} candles — not enough for {splits} folds.[/red]")
        return

    grid = _OPTIMIZE_GRIDS[strategy_name]
    valid = _OPTIMIZE_VALID.get(strategy_name, lambda p: True)
    folds = walk_forward(
        candles, STRATEGIES[strategy_name], grid, n_splits=splits, starting_balance=Decimal(str(balance)), valid=valid
    )
    if not folds:
        console.print("[red]No fold had enough data on both sides — try fewer --splits or a higher --limit.[/red]")
        return

    table = Table(title=f"Walk-forward — {symbol} / {strategy_name} ({len(folds)} folds)")
    for column in ("Test window", "Chosen params", "In-sample PnL", "Out-of-sample trades", "Out-of-sample PnL"):
        table.add_column(column)
    for fold in folds:
        table.add_row(
            f"{fold.test_start:%Y-%m-%d} → {fold.test_end:%Y-%m-%d}",
            ", ".join(f"{k}={v}" for k, v in fold.params.items()),
            f"{fold.in_sample.total_pnl:+.2f}",
            str(fold.out_of_sample.total_trades),
            f"{fold.out_of_sample.total_pnl:+.2f}",
        )
    console.print(table)
    _render_backtest_metrics("Out-of-sample (chained across folds)", walk_forward_summary(folds))
    console.print(
        "[dim]This is the honest number — in-sample PnL above is what optimize-params would have shown you "
        "for that fold and is expected to look better than out-of-sample.[/dim]"
    )


@app.command("train-model")
def train_model_cmd(
    symbol: str = typer.Option(
        ..., "--symbol", help="Symbol(s) to train on, comma-separated, e.g. BTCUSDT,ETHUSDT — pooled into one dataset"
    ),
    provider: str = typer.Option(None, "--provider", help="Data source — overrides config.yaml provider"),
    timeframe: str = typer.Option(None, "--timeframe", help="Overrides config.yaml timeframe"),
    limit: int = typer.Option(3000, "--limit", help="How many historical candles to fetch per symbol — more is better"),
    horizon: int = typer.Option(5, "--horizon", help="Bars ahead to predict: label = will price be higher this many bars from now"),
    test_fraction: float = typer.Option(
        0.25, "--test-fraction", help="Fraction of each symbol's most recent history held out as a chronological out-of-sample test"
    ),
    model_path: str = typer.Option(..., "--model-path", help="Where to save the trained model (.joblib)"),
) -> None:
    """Train the ML strategy's classifier on historical candles and save it.

    `--strategy ml` can't run until you've done this — see ARCHITECTURE.md
    on why training is a separate offline step from the strategy itself.
    Pass multiple comma-separated symbols to pool their history into one
    dataset, so the model learns patterns that generalize across
    instruments instead of one symbol's idiosyncrasies — see
    train_model_multi() in tradon/ml/train.py for how the split stays
    look-ahead-safe when pooling. Requires the `ml` extra:
    `pip install -e .[ml]`.
    """
    symbols_ = [s.strip() for s in symbol.split(",") if s.strip()]
    asyncio.run(_train_model(symbols_, provider, timeframe, limit, horizon, test_fraction, model_path))


async def _train_model(
    symbols: list[str],
    provider_name: str | None,
    timeframe: str | None,
    limit: int,
    horizon: int,
    test_fraction: float,
    model_path: str,
) -> None:
    try:
        from tradon.ml.train import save_model, train_model_multi
    except ImportError as exc:
        raise typer.BadParameter("ML support isn't installed — run `pip install -e .[ml]` first") from exc

    cfg = load_runtime_config()
    if provider_name:
        if provider_name not in PROVIDERS:
            raise typer.BadParameter(f"Unknown provider {provider_name!r} — choices: {', '.join(PROVIDERS)}")
        cfg.provider_name = provider_name
    provider = build_provider(cfg)
    timeframe = timeframe or cfg.timeframe
    symbols = [normalize_symbol(cfg.provider_name, s) for s in symbols]

    candles_by_symbol = {}
    for sym in symbols:
        console.print(f"[dim]Fetching {limit} {timeframe} candles for {sym} from {provider.name}...[/dim]")
        try:
            candles_by_symbol[sym] = await provider.get_ohlcv(sym, timeframe, limit=limit)
        except NotImplementedError as exc:
            raise typer.BadParameter(f"{provider.name} isn't wired up yet: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean message, not a raw traceback
            raise typer.BadParameter(f"Could not fetch {sym} from {provider.name}: {describe_error(exc)}") from exc

    try:
        model, report = train_model_multi(candles_by_symbol, horizon=horizon, test_fraction=test_fraction)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    save_model(model, model_path)

    table = Table(title=f"ML model — {', '.join(symbols)} / {timeframe}, horizon={horizon} bars")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Labeled samples", str(report.samples))
    table.add_row("Train accuracy", f"{report.train_accuracy:.1%}")
    table.add_row("Test accuracy (out-of-sample)", f"{report.test_accuracy:.1%}")
    table.add_row("Test baseline (majority class)", f"{report.test_baseline_accuracy:.1%}")
    console.print(table)
    console.print(f"[green]Saved to {model_path}[/green]")

    if report.test_accuracy <= report.test_baseline_accuracy:
        console.print(
            "[red]Test accuracy did not beat the majority-class baseline — this model has no real "
            "predictive edge on this data. Don't enable it for trading; it's overfit or the features "
            "just aren't informative for this symbol(s)/timeframe.[/red]"
        )
    console.print(
        "[dim]To use it, add to config.yaml's strategies list: name: ml, params: {model_path: "
        f'"{model_path}", long_threshold: 0.55, short_threshold: 0.45}}[/dim]'
    )


def _render_backtest_metrics(title: str, result) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Starting balance", f"{result.starting_balance:.2f}")
    table.add_row("Ending balance", f"{result.ending_balance:.2f}")
    table.add_row("Total PnL", f"{result.total_pnl:+.2f}")
    table.add_row("Win rate", f"{result.win_rate:.1%}" if result.trades else "n/a")
    table.add_row("Profit factor", f"{result.profit_factor:.2f}" if result.profit_factor is not None else "n/a")
    table.add_row("Sharpe ratio", f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "n/a")
    table.add_row("Sortino ratio", f"{result.sortino_ratio:.2f}" if result.sortino_ratio is not None else "n/a")
    table.add_row("Max drawdown", f"{result.max_drawdown:.1%}")
    table.add_row("Wins / Losses", f"{result.wins} / {result.losses}")
    table.add_row("Total trades", str(result.total_trades))
    console.print(table)


def _render_correlation_matrix(candles_by_symbol: dict) -> None:
    symbols = list(candles_by_symbol)
    if len(symbols) < 2:
        return
    matrix = correlation_matrix(candles_by_symbol)
    table = Table(title="Price correlation (close-to-close returns)")
    table.add_column("")
    for symbol in symbols:
        table.add_column(symbol)
    for row_symbol in symbols:
        row = [row_symbol]
        for col_symbol in symbols:
            value = matrix.get(row_symbol, {}).get(col_symbol)
            row.append(f"{value:+.2f}" if value is not None else "n/a")
        table.add_row(*row)
    console.print(table)
    console.print(
        "[dim]Near +1 = moves together (running the same directional strategy on both isn't real "
        "diversification); near -1 = moves opposite; near 0 = unrelated.[/dim]"
    )


def _render(signals) -> None:
    table = Table(title="Signals")
    for column in ("Symbol", "Strategy", "Direction", "Strength", "Size", "Stop", "Target", "Reason"):
        table.add_column(column)
    for sp in signals:
        s, plan = sp.signal, sp.plan
        table.add_row(
            s.symbol,
            s.strategy,
            s.direction.value,
            f"{s.strength:.2f}",
            f"{plan.size:.4f}" if plan else "n/a",
            f"{plan.stop_loss:.5f}" if plan else "n/a",
            f"{plan.take_profit:.5f}" if plan else "n/a",
            s.reason,
        )

    console.print(table)


@app.command("create-user")
def create_user(
    email: str = typer.Option(..., "--email"),
    password: str = typer.Option(..., "--password", prompt=True, hide_input=True, confirmation_prompt=True),
) -> None:
    """Create a Web dashboard login (see ARCHITECTURE.md §2 Этап 4 — no public signup)."""
    asyncio.run(_create_user(email, password))


async def _create_user(email: str, password: str) -> None:
    from tradon.db.models import User
    from tradon.web.auth import get_user_by_email, hash_password

    session_maker = get_sessionmaker()
    async with session_maker() as session:
        if await get_user_by_email(session, email) is not None:
            raise typer.BadParameter(f"User {email!r} already exists")
        session.add(User(email=email, hashed_password=hash_password(password), is_active=True))
        await session.commit()
    await get_engine().dispose()
    console.print(f"[green]Created user {email}[/green]")


if __name__ == "__main__":
    app()
