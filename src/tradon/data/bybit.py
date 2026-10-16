"""Bybit adapter (Stage 2 crypto, see ARCHITECTURE.md §2 Этап 2).

Public market-data endpoints only — no API key needed. v5 API.
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

# Bybit v5 kline "interval": minutes as plain numbers, D/W/M for day/week/month.
_INTERVAL_BY_TIMEFRAME = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}


class BybitProvider(DataProvider):
    name = "bybit"

    def __init__(self, base_url: str = "https://api.bybit.com", category: str = "spot") -> None:
        self._base_url = base_url.rstrip("/")
        self._category = category

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
            resp = await self._get_with_retry(client, "/v5/market/instruments-info", {"category": self._category})
            data = resp.json()["result"]["list"]
            return [s["symbol"] for s in data if s.get("status") == "Trading"]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        interval = _INTERVAL_BY_TIMEFRAME.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Bybit: {timeframe!r}")

        async with self._client() as client:
            resp = await self._get_with_retry(
                client,
                "/v5/market/kline",
                {"category": self._category, "symbol": symbol, "interval": interval, "limit": limit},
            )
            rows = resp.json()["result"]["list"]  # newest first
            return [self._to_candle(symbol, timeframe, row) for row in reversed(rows)]

    async def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        interval = _INTERVAL_BY_TIMEFRAME.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for Bybit: {timeframe!r}")

        ws_url = f"wss://stream.bybit.com/v5/public/{self._category}"
        async with websockets.connect(ws_url, proxy=None) as ws:
            args = [f"kline.{interval}.{s}" for s in symbols]
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            async for raw in ws:
                message = json.loads(raw)
                if message.get("topic", "").startswith("kline."):
                    symbol = message["topic"].split(".")[-1]
                    for row in message["data"]:
                        if row.get("confirm"):  # only closed candles
                            yield self._to_candle(symbol, timeframe, row, is_ws=True)

    @staticmethod
    def _to_candle(symbol: str, timeframe: str, row, is_ws: bool = False) -> Candle:
        if is_ws:
            t, o, h, l, c, v = row["start"], row["open"], row["high"], row["low"], row["close"], row["volume"]
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
            source="bybit",
        )
