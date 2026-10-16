from datetime import UTC, datetime
from decimal import Decimal

from tradon.risk.manager import RiskManager
from tradon.signal_engine.engine import SignalEngine
from tradon.strategy.base import Signal, SignalDirection
from tradon.strategy.indicators import ema, rsi, sma
from tradon.strategy.plugins.rsi_strategy import RsiStrategy
from tradon.strategy.plugins.sma_cross import SmaCrossStrategy


def _d(values: list[float]) -> list[Decimal]:
    return [Decimal(str(v)) for v in values]


def test_sma() -> None:
    assert sma(_d([1, 2, 3, 4, 5]), 3) == Decimal(4)
    assert sma(_d([1, 2]), 3) is None


def test_ema_converges_towards_last_value() -> None:
    values = _d([1] * 10 + [2] * 10)
    result = ema(values, 5)
    assert result is not None
    assert Decimal("1") < result <= Decimal("2")


def test_rsi_monotonic_decline_is_oversold() -> None:
    values = _d([100 - i for i in range(20)])
    result = rsi(values, period=14)
    assert result == Decimal(0)


def test_sma_cross_strategy_emits_long_on_uptrend(candle_factory) -> None:
    closes = [1.0] * 15 + [1.0 + i * 0.05 for i in range(1, 16)]
    candles = candle_factory(closes)
    strategy = SmaCrossStrategy({"fast_period": 5, "slow_period": 10})

    signals = [strategy.on_candles(candles[: i + 1]) for i in range(len(candles))]
    fired = [s for s in signals if s is not None]

    assert any(s.direction is SignalDirection.LONG for s in fired)


def test_rsi_strategy_emits_long_when_oversold(candle_factory) -> None:
    closes = [100 - i for i in range(20)]
    candles = candle_factory(closes)
    strategy = RsiStrategy({"period": 14, "oversold": 30, "overbought": 70})

    signal = strategy.on_candles(candles)

    assert signal is not None
    assert signal.direction is SignalDirection.LONG


def test_risk_manager_long_plan_brackets_entry() -> None:
    manager = RiskManager()
    signal = Signal(
        strategy="sma_cross",
        symbol="EURUSD",
        direction=SignalDirection.LONG,
        strength=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        reason="test",
    )

    plan = manager.plan(
        signal,
        entry_price=Decimal("1.1000"),
        account_balance=Decimal("10000"),
        atr=Decimal("0.0010"),
    )

    assert plan.stop_loss < plan.entry < plan.take_profit
    assert plan.size > 0


async def test_signal_engine_dedups_same_strategy() -> None:
    engine = SignalEngine(min_confirmations=1)
    signal = Signal(
        strategy="sma_cross",
        symbol="EURUSD",
        direction=SignalDirection.LONG,
        strength=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        reason="test",
    )

    first = await engine.ingest(signal)
    second = await engine.ingest(signal)

    assert first is signal
    assert second is None


async def test_signal_engine_requires_confirmations() -> None:
    engine = SignalEngine(min_confirmations=2)
    queue = engine.subscribe()
    base_kwargs = dict(
        symbol="EURUSD",
        direction=SignalDirection.LONG,
        strength=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        reason="test",
    )

    first = await engine.ingest(Signal(strategy="sma_cross", **base_kwargs))
    assert first is None
    assert queue.empty()

    second = await engine.ingest(Signal(strategy="rsi", **base_kwargs))
    assert second is not None
    assert not queue.empty()
