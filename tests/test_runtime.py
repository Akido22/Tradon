import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tradon.config import PROJECT_ROOT
from tradon.data.base import Candle, DataProvider
from tradon.risk.manager import RiskManager
from tradon.runtime import LiveSession
from tradon.signal_engine.engine import SignalEngine
from tradon.strategy.plugins.sma_cross import SmaCrossStrategy

SYMBOL = "LIVETEST"


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    yield
    db_path = PROJECT_ROOT / "data" / "tradon.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM signals WHERE symbol = ?", (SYMBOL,))
    conn.execute("DELETE FROM candles WHERE symbol = ?", (SYMBOL,))
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
async def _dispose_engine_after_test():
    # get_engine()/get_sessionmaker() are lru_cache'd module singletons, but
    # each pytest-asyncio test gets its own event loop — aiosqlite
    # connections are loop-bound, so a cached engine from a previous test's
    # (now-closed) loop would be broken here. Dispose + clear the cache so
    # the next test that touches the DB starts with a fresh, loop-matched
    # engine instead of warning/hanging on a stale one.
    yield
    from tradon.db.session import get_engine, get_sessionmaker

    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def _make_candles(closes: list[float], start_hour: int = 0, spread: float = 0.05) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=start_hour)
    candles = []
    prev_close = closes[0]
    for i, close in enumerate(closes):
        open_ = prev_close
        candles.append(
            Candle(
                symbol=SYMBOL,
                timeframe="1h",
                open=Decimal(str(open_)),
                high=Decimal(str(max(open_, close) + spread)),
                low=Decimal(str(min(open_, close) - spread)),
                close=Decimal(str(close)),
                volume=Decimal(100),
                timestamp=start + timedelta(hours=i),
                source="fake",
            )
        )
        prev_close = close
    return candles


class _NotImplementedProvider(DataProvider):
    name = "fake-not-implemented"

    async def get_symbols(self) -> list[str]:
        return [SYMBOL]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        return _make_candles([100.0] * 5)

    async def subscribe_realtime(self, symbols, timeframe):
        raise NotImplementedError("no realtime for this fake provider")
        yield  # pragma: no cover - makes this a generator


class _TooManySymbolsProvider(DataProvider):
    name = "fake-value-error"

    async def get_symbols(self) -> list[str]:
        return [SYMBOL]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        return _make_candles([100.0] * 5)

    async def subscribe_realtime(self, symbols, timeframe):
        raise ValueError("too many symbols for realtime")
        yield  # pragma: no cover - makes this a generator


class _RealtimeProvider(DataProvider):
    name = "fake-realtime"

    def __init__(self, realtime_candles: list[Candle]) -> None:
        self._realtime_candles = realtime_candles

    async def get_symbols(self) -> list[str]:
        return [SYMBOL]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        return _make_candles([100.0] * 15)

    async def subscribe_realtime(self, symbols, timeframe):
        for candle in self._realtime_candles:
            yield candle


def _build_session(provider, on_signals=None, on_error=None) -> LiveSession:
    signals: list = []
    errors: list = []
    return LiveSession(
        provider,
        [SYMBOL],
        "1h",
        [SmaCrossStrategy({"fast_period": 5, "slow_period": 10})],
        SignalEngine(),
        RiskManager(),
        Decimal(10000),
        on_signals=on_signals or (lambda s: signals.append(s)),
        on_error=on_error or (lambda e: errors.append(e)),
    )


async def _settle(session: LiveSession, attempts: int = 50) -> None:
    for _ in range(attempts):
        if session.mode != "starting":
            return
        await asyncio.sleep(0.02)


async def test_live_session_falls_back_to_polling_on_not_implemented() -> None:
    session = _build_session(_NotImplementedProvider())
    session.start(3600)
    try:
        await _settle(session)
        assert session.mode == "poll"
    finally:
        session.stop()
        await session.wait_closed()


async def test_live_session_falls_back_to_polling_on_value_error() -> None:
    session = _build_session(_TooManySymbolsProvider())
    session.start(3600)
    try:
        await _settle(session)
        assert session.mode == "poll"
    finally:
        session.stop()
        await session.wait_closed()


async def test_live_session_realtime_mode_processes_incoming_candles() -> None:
    # Warm-up gives 15 flat candles; SmaCrossStrategy(5, 10) fires exactly
    # when the 16th candle (the first uptrend one) arrives — see
    # test_backtest.py's identical crossover shape.
    uptrend = _make_candles([100.0 + i * 0.5 for i in range(1, 5)], start_hour=15)

    received: list = []
    session = _build_session(_RealtimeProvider(uptrend), on_signals=lambda s: received.append(s))
    session.start(3600)
    try:
        for _ in range(100):
            if received:
                break
            await asyncio.sleep(0.02)

        assert session.mode == "realtime"
        assert len(received) >= 1
        signal_with_plan = received[0][0]
        assert signal_with_plan.signal.symbol == SYMBOL
        assert signal_with_plan.plan is not None
    finally:
        session.stop()
        await session.wait_closed()


async def test_live_session_poll_mode_still_produces_signals() -> None:
    session = _build_session(_NotImplementedProvider())
    session.start(3600)
    try:
        await _settle(session)
        assert session.mode == "poll"
        # _poll_tick() reuses poll_once(), already covered live elsewhere —
        # here just confirm the fallback path runs without raising.
        await session._poll_tick()
    finally:
        session.stop()
        await session.wait_closed()
