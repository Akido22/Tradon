from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tradon.data.base import Candle


def make_candles(closes: list[float], symbol: str = "EURUSD", timeframe: str = "1h") -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            open=Decimal(str(close)),
            high=Decimal(str(close)),
            low=Decimal(str(close)),
            close=Decimal(str(close)),
            volume=Decimal(100),
            timestamp=start + timedelta(hours=i),
            source="test",
        )
        for i, close in enumerate(closes)
    ]


@pytest.fixture
def candle_factory():
    return make_candles
