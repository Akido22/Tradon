"""MACD signal-line crossover strategy — trend-following, complements sma_cross."""
from __future__ import annotations

from tradon.data.base import Candle
from tradon.strategy.base import Signal, SignalDirection, Strategy
from tradon.strategy.indicators import macd


class MacdStrategy(Strategy):
    name = "macd"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.fast_period = int(self.params.get("fast_period", 12))
        self.slow_period = int(self.params.get("slow_period", 26))
        self.signal_period = int(self.params.get("signal_period", 9))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        min_len = self.slow_period + self.signal_period + 1
        if len(candles) < min_len:
            return None

        closes = [c.close for c in candles]
        prev = macd(closes[:-1], self.fast_period, self.slow_period, self.signal_period)
        now = macd(closes, self.fast_period, self.slow_period, self.signal_period)
        if prev is None or now is None:
            return None
        macd_prev, signal_prev = prev
        macd_now, signal_now = now

        crossed_up = macd_prev <= signal_prev and macd_now > signal_now
        crossed_down = macd_prev >= signal_prev and macd_now < signal_now
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
            reason=(
                f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) "
                f"signal-line crossover ({direction.value})"
            ),
        )
