"""Binance adapter (Stage 2 crypto, see ARCHITECTURE.md §2 Этап 2).

Public market-data endpoints only — no API key needed for OHLCV/symbols.
Same resilience pattern as every other adapter: no system-proxy
auto-detection, retry with backoff on connection failures only.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import websockets

from tradon.data.base import Candle, DataProvider

_INTERVAL_BY_TIMEFRAME = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1w": "1w",
    "1M": "1M",
}


class BinanceProvider(DataProvider):
    name = "binance"

    def __init__(self, base_url: str = "https://api.binance.com") -> None:
        self._base_url = base_url.rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base_url, trust_env=False)

    @staticmethod
    async def _get_with_retry(client: httpx.AsyncClient, path: str, params: dict, attempts: int = 3) -> httpx.Response:
        delay = 0.5
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.get(path, params=params)
                resp.raise_for_status()
                return resp
            except httpx.TransportError:
                if attempt == attempts:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    async def get_symbols(self) -> list[str]:
        async with self._client() as client:
            resp = await self._get_with_retry(client, "/api/v3/exchangeInfo", {})
            data = resp.json()
            return [s["symbol"] for s in data["symbols"] if s.get("status") == "TRADING"]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        interval = _INTERVAL_BY_TIMEFRAME.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Binance: {timeframe!r}")

        async with self._client() as client:
            resp = await self._get_with_retry(
                client, "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit}
            )
            return [self._to_candle(symbol, timeframe, row) for row in resp.json()]

    async def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        interval = _INTERVAL_BY_TIMEFRAME.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Binance: {timeframe!r}")

        streams = "/".join(f"{s.lower()}@kline_{interval}" for s in symbols)
        ws_url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        async with websockets.connect(ws_url, proxy=None) as ws:
            async for raw in ws:
                payload = json.loads(raw)["data"]["k"]
                if not payload["x"]:  # only emit closed candles
                    continue
                yield self._to_candle(payload["s"], timeframe, None, kline_event=payload)

    @staticmethod
    def _to_candle(symbol: str, timeframe: str, row: list | None, kline_event: dict | None = None) -> Candle:
        if kline_event is not None:
            o, h, l, c, v, t = (
                kline_event["o"],
                kline_event["h"],
                kline_event["l"],
                kline_event["c"],
                kline_event["v"],
                kline_event["t"],
            )
        else:
            t, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(l)),
            close=Decimal(str(c)),
            volume=Decimal(str(v)),
            timestamp=datetime.fromtimestamp(int(t) / 1000, tz=UTC),
            source="binance",
        )
