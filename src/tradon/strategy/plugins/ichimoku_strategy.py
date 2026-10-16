"""Ichimoku Kinko Hyo cloud strategy — Tenkan/Kijun crossover confirmed by cloud position."""
from __future__ import annotations

from tradon.data.base import Candle
from tradon.strategy.base import Signal, SignalDirection, Strategy
from tradon.strategy.indicators import ichimoku


class IchimokuStrategy(Strategy):
    name = "ichimoku"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.tenkan_period = int(self.params.get("tenkan_period", 9))
        self.kijun_period = int(self.params.get("kijun_period", 26))
        self.senkou_b_period = int(self.params.get("senkou_b_period", 52))
        self.displacement = int(self.params.get("displacement", 26))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        min_len = self.senkou_b_period + self.displacement + 1
        if len(candles) < min_len:
            return None

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        prev = ichimoku(
            highs[:-1], lows[:-1], self.tenkan_period, self.kijun_period, self.senkou_b_period, self.displacement
        )
        now = ichimoku(highs, lows, self.tenkan_period, self.kijun_period, self.senkou_b_period, self.displacement)
        if prev is None or now is None:
            return None
        tenkan_prev, kijun_prev, _, _ = prev
        tenkan_now, kijun_now, senkou_a_now, senkou_b_now = now

        crossed_up = tenkan_prev <= kijun_prev and tenkan_now > kijun_now
        crossed_down = tenkan_prev >= kijun_prev and tenkan_now < kijun_now
        if not (crossed_up or crossed_down):
            return None

        last = candles[-1]
        cloud_top = max(senkou_a_now, senkou_b_now)
        cloud_bottom = min(senkou_a_now, senkou_b_now)
        # Standard Ichimoku practice: only take a crossover in the direction
        # the cloud confirms — a crossover against the cloud is skipped
        # rather than traded, not flipped to the opposite direction.
        if crossed_up and last.close > cloud_top:
            direction = SignalDirection.LONG
        elif crossed_down and last.close < cloud_bottom:
            direction = SignalDirection.SHORT
        else:
            return None

        return Signal(
            strategy=self.name,
            symbol=last.symbol,
            direction=direction,
            strength=1.0,
            timestamp=last.timestamp,
            reason=f"Ichimoku Tenkan/Kijun crossover confirmed by cloud position ({direction.value})",
        )
