import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tradon.data.base import Candle, DataProvider


def make_candles(closes: list[float], symbol: str = "EURUSD", timeframe: str = "1h") -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=Decimal(str(close)),
            high=Decimal(str(close)),
            low=Decimal(str(close)),
            close=Decimal(str(close)),
            volume=Decimal(100),
            timestamp=start + timedelta(hours=i),
            source="test",
        )
        for i, close in enumerate(closes)
    ]


def make_oscillating_candles(
    symbol: str = "FAKE", n: int = 300, timeframe: str = "1h", spread: float = 0.05
) -> list[Candle]:
    """A wandering-with-drift price path (not flat, unlike make_candles) so
    ATR > 0 and strategies actually produce trades — needed for CLI/Web
    tests that exercise backtest/optimize/walk-forward/portfolio, all of
    which need real high/low range for RiskManager's stop-distance sizing.
    """
    closes = [100.0 + 5 * math.sin(i / 6) + 0.03 * i for i in range(n)]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = []
    prev_close = closes[0]
    for i, close in enumerate(closes):
        open_ = prev_close
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
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


class FakeProvider(DataProvider):
    """A deterministic, network-free DataProvider double for CLI/Web/GUI
    tests — registered into the real PROVIDERS registry (see
    `fake_provider` fixture) so command/endpoint code paths run exactly as
    they would against a real provider, just without the network.
    """

    name = "fake"

    def __init__(self, n: int = 300, supports_realtime: bool = False) -> None:
        self._n = n
        self._supports_realtime = supports_realtime

    async def get_symbols(self) -> list[str]:
        return ["FAKE", "FAKE2"]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        candles = make_oscillating_candles(symbol=symbol, n=self._n, timeframe=timeframe)
        return candles[-limit:]

    async def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        if not self._supports_realtime:
            raise NotImplementedError("fake provider has no realtime stream")
        for candle in make_oscillating_candles(symbol=symbols[0], n=5, timeframe=timeframe):
            yield candle


@pytest.fixture
def candle_factory():
    return make_candles


@pytest.fixture
def fake_provider(monkeypatch):
    """Registers FakeProvider as PROVIDERS["fake"] for the duration of the
    test. PROVIDERS is a single shared dict object imported by name into
    runtime.py/cli/main.py/web/app.py — mutating it here (not rebinding)
    makes "fake" show up through every one of those import sites.
    """
    from tradon.runtime import PROVIDERS

    monkeypatch.setitem(PROVIDERS, "fake", FakeProvider)
    return FakeProvider
