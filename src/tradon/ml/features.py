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

import math
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
    "volume_ratio",
    "volatility_20",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# The slowest input feature (sma_ratio_10_30) needs 30 bars; leave headroom
# for macd (needs 26 + 9) and bollinger (20), so nothing silently degrades.
MIN_CANDLES = 60


def _volatility(closes: list[Decimal], period: int) -> Decimal | None:
    """Population stdev of close-to-close returns over the trailing
    window — realized volatility, distinct from ATR (which is range-based,
    not return-based) and already used by the risk manager, so this gives
    the model an independent read on "how choppy has it been lately".
    """
    if len(closes) < period + 1:
        return None
    returns = []
    for i in range(-period, 0):
        prev = closes[i - 1]
        if prev == 0:
            return None
        returns.append((closes[i] - prev) / prev)
    mean = sum(returns) / Decimal(len(returns))
    variance = sum((r - mean) ** 2 for r in returns) / Decimal(len(returns))
    return variance.sqrt()


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
    volumes = [c.volume for c in candles]

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
    volume_sma = sma(volumes, 20)
    volatility = _volatility(closes, 20)
    if (
        None in (rsi_value, macd_value, sma_fast, sma_slow, atr_value, bands, volume_sma, volatility)
        or sma_slow == 0
    ):
        return None

    macd_line, signal_line = macd_value
    _middle, upper, lower = bands
    band_width = upper - lower
    last_close = closes[-1]
    last_candle = candles[-1]

    # Cyclical (sin/cos) rather than raw hour/weekday, so e.g. 23:00 and
    # 00:00 read as adjacent instead of as far apart as they'd look as
    # plain integers — crypto trades 24/7, so intraday/weekly timing
    # patterns (if any) are at least representable, not an artifact of
    # where the clock happens to wrap.
    hour_angle = 2 * math.pi * last_candle.timestamp.hour / 24
    dow_angle = 2 * math.pi * last_candle.timestamp.weekday() / 7

    return [
        pct_change(1),
        pct_change(5),
        pct_change(10),
        float(rsi_value),
        float(macd_line - signal_line),
        float(sma_fast / sma_slow),
        float(atr_value / last_close) if last_close != 0 else 0.0,
        float((last_close - lower) / band_width) if band_width != 0 else 0.5,
        float(last_candle.volume / volume_sma) if volume_sma != 0 else 1.0,
        float(volatility),
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(dow_angle),
        math.cos(dow_angle),
    ]
