"""Shared orchestration core for every presentation layer (CLI, GUI, ...).

Per ARCHITECTURE.md §3.7: "CLI/GUI/Web/боты не содержат логики анализа — они
только вызывают API ядра и рендерят результат." This module IS that API —
it wires Data/Strategy/Signal layers together and persists results, but
never prints or draws anything. Callers decide how to show what comes back.
"""
from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tradon.scheduler.scheduler import TaskScheduler

from tradon.config import load_app_config
from tradon.data.base import Candle, DataProvider
from tradon.data.binance import BinanceProvider
from tradon.data.bybit import BybitProvider
from tradon.data.fallback import FallbackProvider
from tradon.data.yahoo import YahooProvider
from tradon.db.models import Candle as CandleRow
from tradon.db.models import SignalRecord, StrategyConfig
from tradon.db.session import get_sessionmaker
from tradon.risk.manager import PositionPlan, RiskManager
from tradon.signal_engine.engine import SignalEngine
from tradon.strategy.base import Signal, Strategy
from tradon.strategy.indicators import atr
from tradon.strategy.plugins.atr_breakout_strategy import AtrBreakoutStrategy
from tradon.strategy.plugins.bollinger_strategy import BollingerBandsStrategy
from tradon.strategy.plugins.ichimoku_strategy import IchimokuStrategy
from tradon.strategy.plugins.macd_strategy import MacdStrategy
from tradon.strategy.plugins.rsi_strategy import RsiStrategy
from tradon.strategy.plugins.sma_cross import SmaCrossStrategy

ATR_PERIOD = 14

PROVIDERS: dict[str, type[DataProvider]] = {
    "yahoo": YahooProvider,
    "binance": BinanceProvider,
    "bybit": BybitProvider,
}
try:
    from tradon.data.metatrader import MetaTraderProvider

    PROVIDERS["metatrader"] = MetaTraderProvider
except ImportError:
    pass  # MetaTrader5 package is Windows-only and needs a running terminal
STRATEGIES: dict[str, type[Strategy]] = {
    "sma_cross": SmaCrossStrategy,
    "rsi": RsiStrategy,
    "macd": MacdStrategy,
    "bollinger": BollingerBandsStrategy,
    "atr_breakout": AtrBreakoutStrategy,
    "ichimoku": IchimokuStrategy,
}
try:
    from tradon.strategy.plugins.ml_strategy import MLStrategy

    STRATEGIES["ml"] = MLStrategy
except ImportError:
    pass  # scikit-learn/joblib are an optional extra: pip install -e .[ml]


@dataclass
class RuntimeConfig:
    provider_name: str
    timeframe: str
    instruments: list[str]
    strategy_specs: list[dict] = field(default_factory=list)
    fallback_providers: list[str] = field(default_factory=list)
    poll_interval_seconds: int = 60
    min_confirmations: int = 1
    confirmation_window_minutes: int = 5
    risk_account_balance: Decimal = Decimal(10000)
    risk_per_trade: Decimal = Decimal("0.01")
    risk_atr_multiplier: Decimal = Decimal("1.5")
    risk_reward_ratio: Decimal = Decimal("2")


def load_runtime_config() -> RuntimeConfig:
    config = load_app_config()
    engine_cfg = config.get("signal_engine", {})
    risk_cfg = config.get("risk", {})
    return RuntimeConfig(
        provider_name=config.get("provider", "bybit"),
        timeframe=config.get("timeframe", "1h"),
        instruments=list(config.get("instruments", [])),
        strategy_specs=list(config.get("strategies", [])),
        fallback_providers=list(config.get("fallback_providers", [])),
        poll_interval_seconds=int(config.get("poll_interval_seconds", 60)),
        min_confirmations=int(engine_cfg.get("min_confirmations", 1)),
        confirmation_window_minutes=int(engine_cfg.get("confirmation_window_minutes", 5)),
        risk_account_balance=Decimal(str(risk_cfg.get("account_balance", 10000))),
        risk_per_trade=Decimal(str(risk_cfg.get("risk_per_trade", "0.01"))),
        risk_atr_multiplier=Decimal(str(risk_cfg.get("atr_multiplier", "1.5"))),
        risk_reward_ratio=Decimal(str(risk_cfg.get("reward_risk_ratio", "2"))),
    )


def build_provider(cfg: RuntimeConfig) -> DataProvider:
    """Builds the configured provider, or a FallbackProvider chain over it
    when `cfg.fallback_providers` is non-empty (config.yaml's
    `fallback_providers:` list) — e.g. so Bybit going down doesn't stop
    crypto signals if Binance still has the same symbol.
    """
    primary = PROVIDERS[cfg.provider_name]()
    if not cfg.fallback_providers:
        return primary
    unknown = [name for name in cfg.fallback_providers if name not in PROVIDERS]
    if unknown:
        raise ValueError(f"Unknown fallback_providers: {', '.join(unknown)} — choices: {', '.join(PROVIDERS)}")
    fallbacks = [PROVIDERS[name]() for name in cfg.fallback_providers]
    return FallbackProvider([primary, *fallbacks])


def describe_error(exc: Exception) -> str:
    """str(exc) is empty for some exception types (certain httpx/OSError
    wraps around a connection failure, for instance) — fall back to the
    exception's class name so an error surfaced to a user is never just a
    trailing colon with nothing after it.
    """
    return str(exc) or type(exc).__name__


def normalize_symbol(provider_name: str, symbol: str) -> str:
    """Exchanges (Binance, Bybit) use plain uppercase symbols, so
    typing "eurusd" is forgiven. MetaTrader5 broker symbols often carry a
    case-sensitive suffix (e.g. "EURUSDrfd") that forcing uppercase would
    break — for that provider, only whitespace is stripped.
    """
    symbol = symbol.strip()
    return symbol if provider_name == "metatrader" else symbol.upper()


def build_strategies(cfg: RuntimeConfig) -> list[Strategy]:
    return [
        STRATEGIES[spec["name"]](spec.get("params", {}))
        for spec in cfg.strategy_specs
        if spec.get("enabled", True) and spec["name"] in STRATEGIES
    ]


def build_signal_engine(cfg: RuntimeConfig) -> SignalEngine:
    return SignalEngine(
        min_confirmations=cfg.min_confirmations,
        confirmation_window=timedelta(minutes=cfg.confirmation_window_minutes),
    )


def build_risk_manager(cfg: RuntimeConfig) -> RiskManager:
    return RiskManager(
        risk_per_trade=cfg.risk_per_trade,
        atr_multiplier=cfg.risk_atr_multiplier,
        reward_risk_ratio=cfg.risk_reward_ratio,
    )


@dataclass(frozen=True, slots=True)
class SignalWithPlan:
    signal: Signal
    # None when ATR can't be computed yet (not enough history) — the signal
    # still fires, there's just no size/stop/target suggestion for it.
    plan: PositionPlan | None


async def poll_once(
    provider: DataProvider,
    instruments: list[str],
    timeframe: str,
    strategies: list[Strategy],
    signal_engine: SignalEngine,
    risk_manager: RiskManager,
    account_balance: Decimal,
    session_maker: async_sessionmaker[AsyncSession] | None = None,
    candles_out: dict[str, list[Candle]] | None = None,
) -> list[SignalWithPlan]:
    """Fetch candles, run strategies, persist, and return the signals that were published.

    Every published signal gets a RiskManager suggestion (size/stop/target)
    attached — illustrative sizing against `account_balance`, not a real
    order (signal-only, see ARCHITECTURE.md §0/§6). Entry uses the latest
    close since, unlike a backtest, there's no next bar yet to fill at.

    Pass `candles_out` (an empty dict) to also get the raw candles back per
    symbol — e.g. for a GUI chart — without changing the return type for
    every existing caller.
    """
    session_maker = session_maker or get_sessionmaker()
    published: list[SignalWithPlan] = []

    async with session_maker() as session:
        for symbol in instruments:
            candles = await provider.get_ohlcv(symbol, timeframe)
            if not candles:
                continue
            if candles_out is not None:
                candles_out[symbol] = candles
            await _store_candles(session, candles)

            for strategy in strategies:
                raw_signal = strategy.on_candles(candles)
                if raw_signal is None:
                    continue
                signal = await signal_engine.ingest(raw_signal)
                if signal is None:
                    continue
                plan = _build_plan(candles, signal, risk_manager, account_balance)
                await _store_signal(session, signal, plan)
                published.append(SignalWithPlan(signal, plan))
        await session.commit()

    return published


def _build_plan(
    candles: list[Candle], signal: Signal, risk_manager: RiskManager, account_balance: Decimal
) -> PositionPlan | None:
    current_atr = atr([c.high for c in candles], [c.low for c in candles], [c.close for c in candles], ATR_PERIOD)
    if not current_atr or current_atr <= 0:
        return None
    return risk_manager.plan(signal, candles[-1].close, account_balance, current_atr)


class LiveSession:
    """Drives the live signal pipeline (see ARCHITECTURE.md §3.1/§3.6).

    Prefers realtime push (`provider.subscribe_realtime` — WebSocket) over
    REST polling: it warms up history with one REST call per instrument,
    then tries the provider's realtime stream. If the provider doesn't
    support it (`NotImplementedError`, e.g. Yahoo) or the instrument list
    doesn't fit its realtime limits (`ValueError` — a provider's free tier
    capping how many symbols one WebSocket connection can subscribe to,
    for instance), it falls back to REST polling on
    `interval_seconds` — transparently, so CLI/GUI/Telegram/Web don't need
    to know which mode ended up active; `.mode` exposes it if a caller
    wants to show it. A dropped WebSocket reconnects with backoff instead
    of falling back, since by then realtime is known to work here.

    Realtime candles only arrive when a bar actually closes (the providers
    filter out in-progress bars), so — unlike polling, which re-evaluates
    strategies against the still-forming latest candle every tick — the
    first realtime signal can take up to one full timeframe interval to
    show up even on a healthy connection. `.mode` stays "starting" until
    then; this is expected, not a hang.
    """

    JOB_ID = "poll"
    HISTORY_LIMIT = 500

    def __init__(
        self,
        provider: DataProvider,
        instruments: list[str],
        timeframe: str,
        strategies: list[Strategy],
        signal_engine: SignalEngine,
        risk_manager: RiskManager,
        account_balance: Decimal,
        on_signals: Callable[[list[SignalWithPlan]], Awaitable[None] | None],
        on_error: Callable[[Exception], Awaitable[None] | None] | None = None,
    ) -> None:
        self._provider = provider
        self._instruments = instruments
        self._timeframe = timeframe
        self._strategies = strategies
        self._signal_engine = signal_engine
        self._risk_manager = risk_manager
        self._account_balance = account_balance
        self._on_signals = on_signals
        self._on_error = on_error
        self._scheduler = TaskScheduler()
        self._task: asyncio.Task | None = None
        self._interval_seconds = 60
        self._buffers: dict[str, deque[Candle]] = {}
        self.last_candles: dict[str, list[Candle]] = {}
        self.mode = "starting"  # "starting" | "realtime" | "poll"

    def start(self, interval_seconds: int) -> None:
        self._interval_seconds = interval_seconds
        self._task = asyncio.ensure_future(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        self._scheduler.shutdown()

    async def wait_closed(self) -> None:
        """Await after stop() when you need the background task to have
        fully unwound — e.g. before disposing the DB engine, so a
        cancelled-but-not-yet-finished task isn't still holding a pooled
        connection when dispose() tries to reclaim it.
        """
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def reschedule(self, interval_seconds: int) -> None:
        self._interval_seconds = interval_seconds
        if self.mode == "poll":
            self._scheduler.reschedule(self.JOB_ID, interval_seconds)
        # In realtime mode there's nothing to reschedule — new candles
        # drive everything, not a timer.

    async def _run(self) -> None:
        try:
            await self._warm_up()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - handed to on_error, not swallowed
            await self._report_error(exc)
            return

        try:
            stream = self._provider.subscribe_realtime(self._instruments, self._timeframe)
            first = await stream.__anext__()
        except asyncio.CancelledError:
            raise
        except (NotImplementedError, ValueError):
            self._start_polling()
            return

        self.mode = "realtime"
        await self._handle_candle(first)
        await self._consume_realtime(stream)

    async def _consume_realtime(self, stream) -> None:
        async for candle in stream:
            await self._handle_candle(candle)
        await self._reconnect_realtime()  # generator ended on its own

    async def _reconnect_realtime(self) -> None:
        backoff = 1.0
        while True:
            try:
                async for candle in self._provider.subscribe_realtime(self._instruments, self._timeframe):
                    backoff = 1.0
                    await self._handle_candle(candle)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced, then retried
                await self._report_error(exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _warm_up(self) -> None:
        session_maker = get_sessionmaker()
        async with session_maker() as session:
            for symbol in self._instruments:
                candles = await self._provider.get_ohlcv(symbol, self._timeframe, limit=self.HISTORY_LIMIT)
                self._buffers[symbol] = deque(candles, maxlen=self.HISTORY_LIMIT)
                self.last_candles[symbol] = list(candles)
                if candles:
                    await _store_candles(session, candles)
            await session.commit()

    async def _handle_candle(self, candle: Candle) -> None:
        buf = self._buffers.setdefault(candle.symbol, deque(maxlen=self.HISTORY_LIMIT))
        buf.append(candle)
        candles = list(buf)
        self.last_candles[candle.symbol] = candles

        published: list[SignalWithPlan] = []
        session_maker = get_sessionmaker()
        async with session_maker() as session:
            await _store_candles(session, [candle])
            for strategy in self._strategies:
                raw_signal = strategy.on_candles(candles)
                if raw_signal is None:
                    continue
                signal = await self._signal_engine.ingest(raw_signal)
                if signal is None:
                    continue
                plan = _build_plan(candles, signal, self._risk_manager, self._account_balance)
                await _store_signal(session, signal, plan)
                published.append(SignalWithPlan(signal, plan))
            await session.commit()

        if published:
            await self._call(self._on_signals, published)

    def _start_polling(self) -> None:
        self.mode = "poll"
        self._scheduler.add_periodic(self._poll_tick, self._interval_seconds, self.JOB_ID)
        self._scheduler.start()

    async def _poll_tick(self) -> None:
        # APScheduler swallows job exceptions by default (just logs them) —
        # without this, polling errors would go silent instead of reaching
        # whoever is watching (console, GUI dialog).
        try:
            candles_out: dict[str, list[Candle]] = {}
            signals = await poll_once(
                self._provider,
                self._instruments,
                self._timeframe,
                self._strategies,
                self._signal_engine,
                self._risk_manager,
                self._account_balance,
                candles_out=candles_out,
            )
            self.last_candles = candles_out
        except Exception as exc:  # noqa: BLE001 - handed to on_error, not swallowed
            await self._report_error(exc)
            return

        if signals:
            await self._call(self._on_signals, signals)

    async def _report_error(self, exc: Exception) -> None:
        if self._on_error is None:
            raise exc
        await self._call(self._on_error, exc)

    @staticmethod
    async def _call(fn, arg) -> None:
        result = fn(arg)
        if inspect.isawaitable(result):
            await result


async def _store_candles(session: AsyncSession, candles: list[Candle]) -> None:
    rows = [
        {
            "symbol": c.symbol,
            "timeframe": c.timeframe,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "timestamp": c.timestamp,
            "source": c.source,
        }
        for c in candles
    ]
    # Both Postgres and SQLite support ON CONFLICT DO NOTHING, but via
    # different dialect-specific insert() constructs.
    if session.bind.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    else:
        from sqlalchemy.dialects.postgresql import insert as dialect_insert

    stmt = dialect_insert(CandleRow).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "timeframe", "timestamp", "source"])
    await session.execute(stmt)


async def _store_signal(session: AsyncSession, signal: Signal, plan: PositionPlan | None) -> None:
    result = await session.execute(select(StrategyConfig).where(StrategyConfig.name == signal.strategy))
    strategy_row = result.scalar_one_or_none()
    if strategy_row is None:
        strategy_row = StrategyConfig(name=signal.strategy, params={}, enabled=True)
        session.add(strategy_row)
        await session.flush()

    session.add(
        SignalRecord(
            strategy_id=strategy_row.id,
            symbol=signal.symbol,
            direction=signal.direction.value,
            timestamp=signal.timestamp,
            strength=Decimal(str(round(signal.strength, 4))),
            reason=signal.reason,
            size=plan.size if plan else None,
            entry=plan.entry if plan else None,
            stop_loss=plan.stop_loss if plan else None,
            take_profit=plan.take_profit if plan else None,
        )
    )
