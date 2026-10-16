"""RSI overbought/oversold strategy — one of the two MVP strategies (ARCHITECTURE.md §2 Этап 1)."""
from __future__ import annotations

from decimal import Decimal

from tradon.data.base import Candle
from tradon.strategy.base import Signal, SignalDirection, Strategy
from tradon.strategy.indicators import rsi


class RsiStrategy(Strategy):
    name = "rsi"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.period = int(self.params.get("period", 14))
        self.oversold = Decimal(str(self.params.get("oversold", 30)))
        self.overbought = Decimal(str(self.params.get("overbought", 70)))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        if len(candles) < self.period + 1:
            return None

        closes = [c.close for c in candles]
        value = rsi(closes, self.period)
        if value is None:
            return None

        if value <= self.oversold:
            direction = SignalDirection.LONG
        elif value >= self.overbought:
            direction = SignalDirection.SHORT
        else:
            return None

        last = candles[-1]
        return Signal(
            strategy=self.name,
            symbol=last.symbol,
            direction=direction,
            strength=float(abs(value - 50) / 50),
            timestamp=last.timestamp,
            reason=f"RSI({self.period})={value:.2f} {direction.value}",
        )
