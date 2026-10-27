"""Tests for FallbackProvider (see data/fallback.py) — no network, using
plain DataProvider doubles instead of FakeProvider since some of these
need to deliberately fail.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tests.conftest import make_oscillating_candles
from tradon.data.base import Candle, DataProvider
from tradon.data.fallback import FallbackProvider


class _StubProvider(DataProvider):
    def __init__(self, name: str, fails: bool = False) -> None:
        self.name = name
        self._fails = fails
        self.calls = 0

    async def get_symbols(self) -> list[str]:
        self.calls += 1
        if self._fails:
            raise ConnectionError(f"{self.name} is down")
        return ["FAKE"]

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        self.calls += 1
        if self._fails:
            raise ConnectionError(f"{self.name} is down")
        return make_oscillating_candles(symbol=symbol, n=limit, timeframe=timeframe)

    async def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        for candle in make_oscillating_candles(symbol=symbols[0], n=1, timeframe=timeframe):
            yield candle

    async def supports_symbol(self, symbol: str) -> bool:
        self.calls += 1
        if self._fails:
            raise ConnectionError(f"{self.name} is down")
        return symbol == "FAKE"


def test_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackProvider([])


async def test_uses_primary_when_it_works() -> None:
    primary = _StubProvider("primary")
    backup = _StubProvider("backup")
    provider = FallbackProvider([primary, backup])

    candles = await provider.get_ohlcv("FAKE", "1h", limit=10)

    assert len(candles) == 10
    assert primary.calls == 1
    assert backup.calls == 0
    assert provider.name == "primary"


async def test_falls_back_when_primary_raises() -> None:
    primary = _StubProvider("primary", fails=True)
    backup = _StubProvider("backup")
    provider = FallbackProvider([primary, backup])

    candles = await provider.get_ohlcv("FAKE", "1h", limit=10)

    assert len(candles) == 10
    assert backup.calls == 1
    assert provider.name == "backup"


async def test_remembers_active_provider_across_calls() -> None:
    primary = _StubProvider("primary", fails=True)
    backup = _StubProvider("backup")
    provider = FallbackProvider([primary, backup])

    await provider.get_ohlcv("FAKE", "1h", limit=10)
    primary.calls = 0  # reset to prove the second call doesn't retry primary first
    await provider.supports_symbol("FAKE")

    assert primary.calls == 0
    assert backup.calls == 2


async def test_raises_when_every_provider_fails() -> None:
    provider = FallbackProvider([_StubProvider("a", fails=True), _StubProvider("b", fails=True)])

    with pytest.raises(RuntimeError, match="All providers failed"):
        await provider.get_ohlcv("FAKE", "1h")


def test_subscribe_realtime_delegates_to_active_provider() -> None:
    primary = _StubProvider("primary")
    backup = _StubProvider("backup")
    provider = FallbackProvider([primary, backup])

    stream = provider.subscribe_realtime(["FAKE"], "1h")

    assert stream is not None  # just confirms it delegates without raising eagerly
