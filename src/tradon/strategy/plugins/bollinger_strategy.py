"""Bollinger Bands mean-reversion strategy — same family as rsi, different indicator."""
from __future__ import annotations

from decimal import Decimal

from tradon.data.base import Candle
from tradon.strategy.base import Signal, SignalDirection, Strategy
from tradon.strategy.indicators import bollinger_bands


class BollingerBandsStrategy(Strategy):
    name = "bollinger"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.period = int(self.params.get("period", 20))
        self.std_dev = Decimal(str(self.params.get("std_dev", 2)))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        if len(candles) < self.period:
            return None

        closes = [c.close for c in candles]
        bands = bollinger_bands(closes, self.period, self.std_dev)
        if bands is None:
            return None
        middle, upper, lower = bands

        last = candles[-1]
        if last.close <= lower:
            direction = SignalDirection.LONG
        elif last.close >= upper:
            direction = SignalDirection.SHORT
        else:
            return None

        half_width = (upper - lower) / 2
        strength = float(min(abs(last.close - middle) / half_width, Decimal(1))) if half_width > 0 else 0.0
        return Signal(
            strategy=self.name,
            symbol=last.symbol,
            direction=direction,
            strength=strength,
            timestamp=last.timestamp,
            reason=(
                f"BB({self.period},{self.std_dev}) close={last.close} "
                f"outside [{lower:.5f}, {upper:.5f}] ({direction.value})"
            ),
        )
