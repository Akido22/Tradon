"""Strategy Engine contract (see ARCHITECTURE.md §3.2).

A Strategy consumes a stream of candles and produces Signal objects.
Strategies are plugins registered by name in config — the core never
changes when a new strategy is added.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradon.data.base import Candle


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True, slots=True)
class Signal:
    strategy: str
    symbol: str
    direction: SignalDirection
    strength: float  # 0.0..1.0
    timestamp: datetime
    reason: str


class Strategy(ABC):
    """Base class every strategy plugin must implement."""

    name: str

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def on_candles(self, candles: list[Candle]) -> Signal | None:
        """Evaluate the latest candle window and optionally emit a Signal."""
