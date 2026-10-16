"""ATR-filtered channel breakout strategy — trend-following, complements sma_cross/macd."""
from __future__ import annotations

from decimal import Decimal

from tradon.data.base import Candle
from tradon.strategy.base import Signal, SignalDirection, Strategy
from tradon.strategy.indicators import atr


class AtrBreakoutStrategy(Strategy):
    name = "atr_breakout"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.channel_period = int(self.params.get("channel_period", 20))
        self.atr_period = int(self.params.get("atr_period", 14))
        self.atr_multiplier = Decimal(str(self.params.get("atr_multiplier", 1)))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        min_len = max(self.channel_period, self.atr_period) + 2
        if len(candles) < min_len:
            return None

        # The channel/ATR are computed on everything BEFORE the last bar —
        # the last bar's own close is what we're testing against them, so
        # including it in the channel would make a breakout compare a bar
        # to a range that already contains it.
        prior = candles[:-1]
        window = prior[-self.channel_period :]
        highest_high = max(c.high for c in window)
        lowest_low = min(c.low for c in window)
        current_atr = atr(
            [c.high for c in prior], [c.low for c in prior], [c.close for c in prior], self.atr_period
        )
        if current_atr is None or current_atr <= 0:
            return None

        last = candles[-1]
        upper = highest_high + self.atr_multiplier * current_atr
        lower = lowest_low - self.atr_multiplier * current_atr
        if last.close > upper:
            direction = SignalDirection.LONG
            breakout_distance = last.close - upper
        elif last.close < lower:
            direction = SignalDirection.SHORT
            breakout_distance = lower - last.close
        else:
            return None

        strength = float(min(breakout_distance / current_atr, Decimal(1)))
        return Signal(
            strategy=self.name,
            symbol=last.symbol,
            direction=direction,
            strength=strength,
            timestamp=last.timestamp,
            reason=(
                f"ATR breakout: close={last.close} broke {self.channel_period}-bar "
                f"{'high' if direction is SignalDirection.LONG else 'low'} by "
                f"{self.atr_multiplier}xATR({self.atr_period}) ({direction.value})"
            ),
        )
