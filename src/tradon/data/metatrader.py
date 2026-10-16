"""MetaTrader5 adapter — reserve Forex source for the MVP (see ARCHITECTURE.md §3.1).

Note: the `MetaTrader5` package only works on Windows (§6 risk) AND requires
a running, logged-in MT5 terminal on the same machine — it talks to the
terminal via local IPC, not the network. All calls are synchronous, so they
run in a worker thread. Verified live against a real broker terminal
(get_symbols, get_ohlcv, subscribe_realtime, and a full backtest all ran
against real data) — broker symbols can carry a case-sensitive prefix/suffix
(e.g. Alfa-Forex's "EURUSDrfd" for plain "EURUSD"), so `_resolve_symbol_sync`
looks up the broker's actual name for whatever plain symbol you pass in and
caches the mapping; you don't need to know or type the decorated name.

The Python API has no push/callback mechanism, so `subscribe_realtime`
polls and only yields once a bar actually closes (see the comment inside
for why that isn't simply `copy_rates_from_pos(..., 0, 1)`).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import MetaTrader5 as mt5

from tradon.data.base import Candle, DataProvider

_TIMEFRAME_BY_STR = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "1d": mt5.TIMEFRAME_D1,
    "1w": mt5.TIMEFRAME_W1,
    "1M": mt5.TIMEFRAME_MN1,
}


class MetaTraderProvider(DataProvider):
    name = "metatrader"

    def __init__(self, poll_seconds: float = 5.0) -> None:
        self._poll_seconds = poll_seconds
        self._symbol_cache: dict[str, str] = {}

    async def get_symbols(self) -> list[str]:
        return await asyncio.to_thread(self._get_symbols_sync)

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        mt5_timeframe = self._resolve_timeframe(timeframe)
        broker_symbol = await asyncio.to_thread(self._resolve_symbol_sync, symbol)
        rates = await asyncio.to_thread(self._copy_rates_sync, broker_symbol, mt5_timeframe, limit)
        return [self._to_candle(symbol, timeframe, row) for row in rates]

    async def subscribe_realtime(self, symbols: list[str], timeframe: str) -> AsyncIterator[Candle]:
        # copy_rates_from_pos's position 0 is the currently forming bar
        # (confirmed live: it returns the just-opened bar's partial OHLC
        # immediately, not a closed one). Fetch 2 bars instead and use
        # index 0 — the bar right before the forming one, i.e. the last
        # one that actually closed — same closed-only semantics as the
        # Binance/Bybit realtime streams.
        mt5_timeframe = self._resolve_timeframe(timeframe)
        broker_symbols = {symbol: await asyncio.to_thread(self._resolve_symbol_sync, symbol) for symbol in symbols}
        last_closed_time: dict[str, int] = {}
        while True:
            for symbol, broker_symbol in broker_symbols.items():
                rows = await asyncio.to_thread(self._copy_rates_sync, broker_symbol, mt5_timeframe, 2)
                if len(rows) < 2:
                    continue
                closed_row = rows[0]
                if last_closed_time.get(symbol) == closed_row["time"]:
                    continue
                last_closed_time[symbol] = closed_row["time"]
                yield self._to_candle(symbol, timeframe, closed_row)
            await asyncio.sleep(self._poll_seconds)

    async def supports_symbol(self, symbol: str) -> bool:
        try:
            await asyncio.to_thread(self._resolve_symbol_sync, symbol)
        except ConnectionError:
            return False
        return True

    @staticmethod
    def _resolve_timeframe(timeframe: str) -> int:
        mt5_timeframe = _TIMEFRAME_BY_STR.get(timeframe)
        if mt5_timeframe is None:
            raise ValueError(f"Unsupported timeframe for MetaTrader5: {timeframe!r}")
        return mt5_timeframe

    @staticmethod
    def _ensure_connected() -> None:
        if not mt5.initialize():
            raise ConnectionError(f"MetaTrader5.initialize() failed: {mt5.last_error()}")

    @classmethod
    def _get_symbols_sync(cls) -> list[str]:
        cls._ensure_connected()
        symbols = mt5.symbols_get()
        if symbols is None:
            raise ConnectionError(f"MetaTrader5.symbols_get() failed: {mt5.last_error()}")
        return [s.name for s in symbols]

    def _resolve_symbol_sync(self, symbol: str) -> str:
        """Map a plain instrument name (e.g. "EURUSD") to whatever the
        connected broker actually calls it. Brokers routinely wrap the plain
        name in their own prefix/suffix — confirmed live against a real
        Alfa-Forex MT5 terminal, which lists "EURUSDrfd" and has no plain
        "EURUSD" at all. Tries the exact name first (cheap, and correct for
        brokers that don't decorate symbols), then falls back to the
        broker's full symbol list and picks whichever name contains the
        requested one as a substring, closest in length (so "EURUSDrfd"
        wins over a longer coincidental match). Cached per instance since
        `symbols_get()` is a full round trip to the terminal.
        """
        cached = self._symbol_cache.get(symbol)
        if cached is not None:
            return cached
        self._ensure_connected()
        if mt5.symbol_info(symbol) is not None:
            self._symbol_cache[symbol] = symbol
            return symbol

        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            raise ConnectionError(f"MetaTrader5.symbols_get() failed: {mt5.last_error()}")
        target = symbol.upper()
        candidates = [s.name for s in all_symbols if target in s.name.upper()]
        if not candidates:
            raise ConnectionError(
                f"No MetaTrader5 symbol matches {symbol!r} for this broker — "
                f"check the exact name in MetaTrader5.symbols_get() (e.g. a broker prefix/suffix)"
            )
        resolved = min(candidates, key=lambda name: (len(name) - len(target), name))
        mt5.symbol_select(resolved, True)
        self._symbol_cache[symbol] = resolved
        return resolved

    @classmethod
    def _copy_rates_sync(cls, symbol: str, mt5_timeframe: int, count: int) -> list:
        cls._ensure_connected()
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        if rates is None:
            raise ConnectionError(f"MetaTrader5.copy_rates_from_pos({symbol!r}) failed: {mt5.last_error()}")
        return list(rates)

    @staticmethod
    def _to_candle(symbol: str, timeframe: str, row) -> Candle:
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=Decimal(str(row["tick_volume"])),
            timestamp=datetime.fromtimestamp(int(row["time"]), tz=UTC),
            source="metatrader",
        )
