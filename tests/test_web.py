from datetime import UTC, datetime
from decimal import Decimal

from tradon.risk.manager import PositionPlan
from tradon.runtime import SignalWithPlan
from tradon.strategy.base import Signal, SignalDirection
from tradon.web.app import _signal_to_json


def _signal() -> Signal:
    return Signal(
        strategy="sma_cross",
        symbol="BTCUSDT",
        direction=SignalDirection.LONG,
        strength=0.9,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        reason="SMA10/30 crossover (long)",
    )


def test_signal_to_json_includes_plan_fields() -> None:
    plan = PositionPlan(
        symbol="BTCUSDT",
        direction=SignalDirection.LONG,
        size=Decimal("1.2345"),
        entry=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
    )

    payload = _signal_to_json(SignalWithPlan(signal=_signal(), plan=plan))

    assert payload["type"] == "signal"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["direction"] == "long"
    assert payload["size"] == 1.2345
    assert payload["stop_loss"] == 49000.0
    assert payload["take_profit"] == 52000.0


def test_signal_to_json_handles_missing_plan() -> None:
    payload = _signal_to_json(SignalWithPlan(signal=_signal(), plan=None))

    assert payload["size"] is None
    assert payload["stop_loss"] is None
    assert payload["take_profit"] is None
