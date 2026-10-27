"""Multi-provider fallback wrapper (see ARCHITECTURE.md §3.2's reliability
concerns for live trading: a single exchange being down/rate-limited
shouldn't silently stop the whole signal pipeline when another provider
could serve the same symbol).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from tradon.data.base import Candle, DataProvider


def _describe(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


class FallbackProvider(DataProvider):
    """Tries `providers` in order for every REST call, remembering which one
    last worked so the next call starts there instead of always retrying a
    known-dead provider first. Only wraps the REST path (`get_symbols`,
    `get_ohlcv`, `supports_symbol`) — `subscribe_realtime` just delegates to
    the currently active provider as-is. Switching providers mid-stream
    would mean re-warming history from a different source, which
    `LiveSession` isn't set up to do; its existing realtime-to-poll
    fallback already covers a provider whose realtime feed doesn't work at
    all, so a REST-level failure on the poll path is what this adds
    coverage for.
    """

    def __init__(self, providers: list[DataProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = providers
        self._active_index = 0

    @property
    def name(self) -> str:
        return self._providers[self._active_index].name

    async def get_symbols(self) -> list[str]:
        return await self._call(lambda p: p.get_symbols())

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        return await self._call(lambda p: p.get_ohlcv(symbol, timeframe, limit))

    async def supports_symbol(self, symbol: str) -> bool:
        return await self._call(lambda p: p.supports_symbol(symbol))

    def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        return self._providers[self._active_index].subscribe_realtime(symbols, timeframe)

    async def _call(self, fn):
        errors = []
        for offset in range(len(self._providers)):
            index = (self._active_index + offset) % len(self._providers)
            provider = self._providers[index]
            try:
                result = await fn(provider)
            except Exception as exc:  # noqa: BLE001 - tries the rest before giving up
                errors.append(f"{provider.name}: {_describe(exc)}")
                continue
            self._active_index = index
            return result
        raise RuntimeError(f"All providers failed: {'; '.join(errors)}")
