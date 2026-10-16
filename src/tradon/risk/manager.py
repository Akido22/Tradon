"""Risk Management (see ARCHITECTURE.md §3.4).

Independent of any specific strategy: takes a Signal plus risk parameters
and computes position size / stop-loss / take-profit.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from tradon.strategy.base import Signal, SignalDirection


@dataclass(frozen=True, slots=True)
class PositionPlan:
    symbol: str
    direction: SignalDirection
    size: Decimal
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal


class RiskManager:
    def __init__(
        self,
        risk_per_trade: Decimal = Decimal("0.01"),
        atr_multiplier: Decimal = Decimal("1.5"),
        reward_risk_ratio: Decimal = Decimal("2"),
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.reward_risk_ratio = reward_risk_ratio

    def plan(
        self,
        signal: Signal,
        entry_price: Decimal,
        account_balance: Decimal,
        atr: Decimal,
    ) -> PositionPlan:
        stop_distance = atr * self.atr_multiplier
        risk_amount = account_balance * self.risk_per_trade
        size = risk_amount / stop_distance if stop_distance > 0 else Decimal(0)

        if signal.direction is SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * self.reward_risk_ratio
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self.reward_risk_ratio

        return PositionPlan(
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
