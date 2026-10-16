"""Feature engineering for the ML strategy (see ARCHITECTURE.md — ML is an
optional extra, `pip install -e .[ml]`).

Turns a trailing candle window into a fixed-size numeric vector, reusing
the same indicator library every other strategy plugin uses rather than a
separate ML-specific implementation. Shared by training
(tradon.ml.train.build_dataset) and inference (MLStrategy.on_candles) so
the two can never drift out of sync with each other — training on one
feature layout and predicting with another would silently produce garbage.
"""
from __future__ import annotations

from decimal import Decimal

from tradon.data.base import Candle
from tradon.strategy.indicators import atr, bollinger_bands, macd, rsi, sma

FEATURE_NAMES = [
    "return_1",
    "return_5",
    "return_10",
    "rsi_14",
    "macd_hist",
    "sma_ratio_10_30",
    "atr_pct",
    "bb_position",
]

# The slowest input feature (sma_ratio_10_30) needs 30 bars; leave headroom
# for macd (needs 26 + 9) and bollinger (20), so nothing silently degrades.
MIN_CANDLES = 60


def extract_features(candles: list[Candle]) -> list[float] | None:
    """Compute FEATURE_NAMES describing the state as of the LAST candle in
    `candles` (chronological, oldest first) — i.e. what would actually be
    known right before deciding whether to enter a trade on the next bar.
    Returns None if there isn't enough history yet, same warm-up
    convention as every indicator-based strategy.
    """
    if len(candles) < MIN_CANDLES:
        return None

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    def pct_change(n: int) -> float:
        base = closes[-1 - n]
        if base == 0:
            return 0.0
        return float((closes[-1] - base) / base)

    rsi_value = rsi(closes, 14)
    macd_value = macd(closes, 12, 26, 9)
    sma_fast = sma(closes, 10)
    sma_slow = sma(closes, 30)
    atr_value = atr(highs, lows, closes, 14)
    bands = bollinger_bands(closes, 20, Decimal(2))
    if None in (rsi_value, macd_value, sma_fast, sma_slow, atr_value, bands) or sma_slow == 0:
        return None

    macd_line, signal_line = macd_value
    _middle, upper, lower = bands
    band_width = upper - lower
    last_close = closes[-1]

    return [
        pct_change(1),
        pct_change(5),
        pct_change(10),
        float(rsi_value),
        float(macd_line - signal_line),
        float(sma_fast / sma_slow),
        float(atr_value / last_close) if last_close != 0 else 0.0,
        float((last_close - lower) / band_width) if band_width != 0 else 0.5,
    ]
