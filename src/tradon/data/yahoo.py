"""Yahoo Finance adapter — reserve Forex source for the MVP (see ARCHITECTURE.md §3.1).

`yfinance` is synchronous, so every call runs in a worker thread
(`asyncio.to_thread`) to keep the DataProvider interface async. No API key,
but no realtime push API either — Yahoo has no forex symbol-discovery
endpoint, so `get_symbols()` returns a curated list of majors.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC
from decimal import Decimal

import yfinance as yf

from tradon.data.base import Candle, DataProvider

_MAJORS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "USDCNH", "XAUUSD",
]

_INTERVAL_BY_TIMEFRAME = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
}
_INTRADAY_PERIOD = "7d"
_DAILY_PERIOD = "2y"


class YahooProvider(DataProvider):
    name = "yahoo"

    async def get_symbols(self) -> list[str]:
        return list(_MAJORS)

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        interval = _INTERVAL_BY_TIMEFRAME.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Yahoo: {timeframe!r}")
        period = _INTRADAY_PERIOD if timeframe in ("1m", "5m", "15m", "30m", "1h") else _DAILY_PERIOD

        # Yahoo's undocumented API is flaky — it fails intermittently with a
        # "possibly delisted" warning printed straight to the console (not
        # an exception) and an empty DataFrame. Retry a couple of times
        # before giving up so one transient blip doesn't skip a whole poll.
        delay = 1.0
        for attempt in range(1, 4):
            df = await asyncio.to_thread(self._fetch_history, symbol, interval, period)
            if not df.empty:
                break
            if attempt < 3:
                await asyncio.sleep(delay)
                delay *= 2

        return [self._to_candle(symbol, timeframe, row) for row in df.tail(limit).itertuples()]

    async def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        raise NotImplementedError("Yahoo has no realtime push API")
        yield  # pragma: no cover - makes this an async generator

    @staticmethod
    def _fetch_history(symbol: str, interval: str, period: str):
        ticker = yf.Ticker(f"{symbol}=X")
        return ticker.history(period=period, interval=interval)

    @staticmethod
    def _to_candle(symbol: str, timeframe: str, row) -> Candle:
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=Decimal(str(row.Open)),
            high=Decimal(str(row.High)),
            low=Decimal(str(row.Low)),
            close=Decimal(str(row.Close)),
            volume=Decimal(str(row.Volume)),
            timestamp=row.Index.to_pydatetime().astimezone(UTC),
            source="yahoo",
        )
