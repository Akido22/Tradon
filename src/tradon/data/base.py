"""Unified Data Layer contract: Candle model + DataProvider interface.

Every adapter (Yahoo, MetaTrader, Binance, Bybit) normalizes to this
model so the rest of the core never has to know which source produced the data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime  # timezone-aware, UTC
    source: str


class DataProvider(ABC):
    """Interface every market data adapter must implement."""

    name: str

    @abstractmethod
    async def get_symbols(self) -> list[str]:
        """Return the list of symbols available from this provider."""

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[Candle]:
        """Fetch historical OHLCV candles (REST)."""

    @abstractmethod
    def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        """Stream realtime candles/ticks (WebSocket) for the given symbols."""

    async def supports_symbol(self, symbol: str) -> bool:
        """Whether this provider can serve `symbol`. Default is exact
        membership in get_symbols() — right for exchanges, where the
        instrument name IS the API symbol. MetaTraderProvider overrides
        this because a broker's actual symbol name can differ from the
        plain name callers pass in (see its _resolve_symbol_sync)."""
        return symbol in set(await self.get_symbols())
