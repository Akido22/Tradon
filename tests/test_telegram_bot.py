from datetime import UTC, datetime
from decimal import Decimal

from tradon.bots.telegram import format_signal
from tradon.risk.manager import PositionPlan
from tradon.runtime import SignalWithPlan
from tradon.strategy.base import Signal, SignalDirection


def _signal() -> Signal:
    return Signal(
        strategy="rsi",
        symbol="EURUSD",
        direction=SignalDirection.SHORT,
        strength=0.72,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        reason="RSI(14)=74.20 short",
    )


def test_format_signal_includes_key_fields() -> None:
    text = format_signal(SignalWithPlan(signal=_signal(), plan=None))

    assert "EURUSD" in text
    assert "rsi" in text
    assert "SHORT" in text
    assert "0.72" in text
    assert "RSI(14)=74.20 short" in text


def test_format_signal_includes_risk_plan_when_present() -> None:
    plan = PositionPlan(
        symbol="EURUSD",
        direction=SignalDirection.SHORT,
        size=Decimal("491.23"),
        entry=Decimal("1.10000"),
        stop_loss=Decimal("1.10500"),
        take_profit=Decimal("1.09000"),
    )

    text = format_signal(SignalWithPlan(signal=_signal(), plan=plan))

    assert "491.2300" in text
    assert "1.10500" in text
    assert "1.09000" in text


def test_format_signal_omits_plan_section_when_none() -> None:
    text = format_signal(SignalWithPlan(signal=_signal(), plan=None))

    assert "Suggested" not in text
