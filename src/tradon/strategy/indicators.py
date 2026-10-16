"""Reusable indicator library shared by strategy plugins (see ARCHITECTURE.md §3.2)."""
from __future__ import annotations

from decimal import Decimal


def sma(values: list[Decimal], period: int) -> Decimal | None:
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / Decimal(period)


def ema(values: list[Decimal], period: int) -> Decimal | None:
    if len(values) < period:
        return None
    k = Decimal(2) / Decimal(period + 1)
    result = sum(values[:period]) / Decimal(period)
    for value in values[period:]:
        result = value * k + result * (Decimal(1) - k)
    return result


def atr(highs: list[Decimal], lows: list[Decimal], closes: list[Decimal], period: int = 14) -> Decimal | None:
    """Average True Range — feeds RiskManager's stop/target distance."""
    if len(closes) < period + 1:
        return None
    true_ranges = []
    for i in range(-period, 0):
        high, low, prev_close = highs[i], lows[i], closes[i - 1]
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(true_ranges) / Decimal(period)


def _ema_series(values: list[Decimal], period: int) -> list[Decimal | None]:
    """EMA at every index of `values` in one forward pass — same seed/anchor
    rule as ema() (SMA of the first `period` values, then rolled forward),
    just without recomputing from scratch for each prefix.
    """
    if len(values) < period:
        return [None] * len(values)
    k = Decimal(2) / Decimal(period + 1)
    series: list[Decimal | None] = [None] * (period - 1)
    current = sum(values[:period]) / Decimal(period)
    series.append(current)
    for value in values[period:]:
        current = value * k + current * (Decimal(1) - k)
        series.append(current)
    return series


def macd(
    values: list[Decimal], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> tuple[Decimal, Decimal] | None:
    """(MACD line, signal line) at the last point of `values`. MACD line is
    fast EMA minus slow EMA; the signal line is an EMA of the MACD line
    itself. Needs roughly slow_period + signal_period points to warm up.
    """
    if len(values) < slow_period + signal_period:
        return None
    fast_series = _ema_series(values, fast_period)
    slow_series = _ema_series(values, slow_period)
    macd_line = [f - s for f, s in zip(fast_series, slow_series, strict=True) if f is not None and s is not None]
    if len(macd_line) < signal_period:
        return None
    signal_series = _ema_series(macd_line, signal_period)
    if signal_series[-1] is None:
        return None
    return macd_line[-1], signal_series[-1]


def bollinger_bands(
    values: list[Decimal], period: int = 20, std_dev: Decimal = Decimal(2)
) -> tuple[Decimal, Decimal, Decimal] | None:
    """(middle, upper, lower) bands: a period-SMA of `values` plus/minus
    `std_dev` population standard deviations of the same trailing window.
    """
    if len(values) < period:
        return None
    window = values[-period:]
    middle = sum(window) / Decimal(period)
    variance = sum((v - middle) ** 2 for v in window) / Decimal(period)
    spread = variance.sqrt() * std_dev
    return middle, middle + spread, middle - spread


def ichimoku(
    highs: list[Decimal],
    lows: list[Decimal],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    """(tenkan, kijun, senkou_a, senkou_b) as they'd actually line up
    against the LAST bar. Tenkan/Kijun are the usual "midpoint of the
    N-bar high/low channel" computed as of now. Senkou A/B are different:
    on a real Ichimoku chart they're plotted `displacement` bars into the
    future, so the cloud edge lined up with TODAY was actually computed
    `displacement` bars ago — using today's data for them instead would
    leak information the cloud wouldn't have had at the time. This
    reproduces that by looking back `displacement` bars first.
    """
    if len(highs) < kijun_period:
        return None
    tenkan = (max(highs[-tenkan_period:]) + min(lows[-tenkan_period:])) / 2
    kijun = (max(highs[-kijun_period:]) + min(lows[-kijun_period:])) / 2

    idx = len(highs) - displacement  # bar index the current cloud value was computed from
    if idx < senkou_b_period:
        return None
    past_highs, past_lows = highs[:idx], lows[:idx]
    past_tenkan = (max(past_highs[-tenkan_period:]) + min(past_lows[-tenkan_period:])) / 2
    past_kijun = (max(past_highs[-kijun_period:]) + min(past_lows[-kijun_period:])) / 2
    senkou_a = (past_tenkan + past_kijun) / 2
    senkou_b = (max(past_highs[-senkou_b_period:]) + min(past_lows[-senkou_b_period:])) / 2
    return tenkan, kijun, senkou_a, senkou_b


def rsi(values: list[Decimal], period: int = 14) -> Decimal | None:
    if len(values) < period + 1:
        return None
    gains = Decimal(0)
    losses = Decimal(0)
    for i in range(-period, 0):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return Decimal(100)
    avg_gain = gains / Decimal(period)
    avg_loss = losses / Decimal(period)
    rs = avg_gain / avg_loss
    return Decimal(100) - (Decimal(100) / (Decimal(1) + rs))
