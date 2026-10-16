"""SMA crossover strategy — one of the two MVP strategies (ARCHITECTURE.md §2 Этап 1)."""
from __future__ import annotations

from tradon.data.base import Candle
from tradon.strategy.base import Signal, SignalDirection, Strategy
from tradon.strategy.indicators import sma


class SmaCrossStrategy(Strategy):
    name = "sma_cross"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.fast_period = int(self.params.get("fast_period", 10))
        self.slow_period = int(self.params.get("slow_period", 30))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        if len(candles) < self.slow_period + 1:
            return None

        closes = [c.close for c in candles]
        fast_prev = sma(closes[:-1], self.fast_period)
        slow_prev = sma(closes[:-1], self.slow_period)
        fast_now = sma(closes, self.fast_period)
        slow_now = sma(closes, self.slow_period)
        if None in (fast_prev, slow_prev, fast_now, slow_now):
            return None

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        if not (crossed_up or crossed_down):
            return None

        last = candles[-1]
        direction = SignalDirection.LONG if crossed_up else SignalDirection.SHORT
        return Signal(
            strategy=self.name,
            symbol=last.symbol,
            direction=direction,
            strength=1.0,
            timestamp=last.timestamp,
            reason=f"SMA{self.fast_period}/{self.slow_period} crossover ({direction.value})",
        )
