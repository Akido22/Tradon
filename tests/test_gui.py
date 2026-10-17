"""GUI tests — optional extra (pip install -e .[gui]), skipped entirely
when PySide6/qasync aren't installed.

Deliberately scoped to construction + synchronous method calls only, never
to running the qasync event loop: driving the full async start/stop
monitoring flow inside an automated offscreen test proved flaky in this
same codebase during manual exploration (asyncio task re-entrancy errors
specific to the test harness, not the app) — that risk isn't worth taking
inside CI, where flaky != useful. `asyncio.ensure_future(self._load_history())`
in MainWindow.__init__ still needs a *set* event loop to bind to (not a
*running* one) — see the qt_app fixture — but the Task it creates is never
actually executed since nothing here calls run_forever()/run_until_complete().
"""
from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("qasync")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import qasync  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tradon.data.base import Candle  # noqa: E402
from tradon.gui.main_window import _STRATEGY_PARAM_SPECS, MainWindow  # noqa: E402
from tradon.runtime import STRATEGIES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    loop = qasync.QEventLoop(app)
    qasync.asyncio.set_event_loop(loop)
    yield app
    loop.close()


@pytest.fixture
def window(qt_app, fake_provider, tmp_path):
    # An isolated, file-backed settings scope — never this machine's real
    # saved GUI preferences (see MainWindow's settings= param docstring).
    settings = QSettings(str(tmp_path / "test_settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    yield win
    win.close()
    win.deleteLater()
    # __init__ and close() each schedule a background task
    # (_load_history/_shutdown) on the qasync loop — inert here since
    # nothing in this module ever runs the loop. cancel() alone only
    # *requests* cancellation; the task is still "pending" at GC time
    # (logging "Task was destroyed but it is pending!") unless something
    # actually drives it to completion — a single run_until_complete over
    # already-cancelled, never-started tasks does that with no risk of the
    # task re-entrancy issues a *running*, multi-task loop hit elsewhere.
    loop = qasync.asyncio.get_event_loop()
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))


def _make_candle(close: float, ts: datetime, symbol: str = "FAKE") -> Candle:
    return Candle(
        symbol=symbol,
        timeframe="1h",
        open=Decimal(str(close)),
        high=Decimal(str(close + 0.5)),
        low=Decimal(str(close - 0.5)),
        close=Decimal(str(close)),
        volume=Decimal(100),
        timestamp=ts,
        source="test",
    )


def test_window_builds_with_configured_instruments_checked(window) -> None:
    assert window._instrument_boxes, "expected at least one instrument checkbox from config.yaml"
    for symbol, box in window._instrument_boxes.items():
        assert box.text() == symbol
        assert box.isChecked()


def test_strategy_checkboxes_cover_every_registered_strategy(window) -> None:
    assert set(window._strategy_boxes) == set(STRATEGIES)


def test_add_pair_creates_a_new_checkbox(window) -> None:
    before = set(window._instrument_boxes)

    window._new_symbol_edit.setText("NEWPAIR")
    window._on_add_pair_clicked()

    assert "NEWPAIR" in window._instrument_boxes
    assert window._instrument_boxes["NEWPAIR"].isChecked()
    assert set(window._instrument_boxes) == before | {"NEWPAIR"}


def test_add_pair_on_existing_symbol_does_not_duplicate(window) -> None:
    window._new_symbol_edit.setText("DUPTEST")
    window._on_add_pair_clicked()
    window._instrument_boxes["DUPTEST"].setChecked(False)

    window._new_symbol_edit.setText("DUPTEST")
    window._on_add_pair_clicked()

    # Re-adding re-checks the existing box rather than creating a second one.
    assert list(window._instrument_boxes).count("DUPTEST") == 1
    assert window._instrument_boxes["DUPTEST"].isChecked()


def test_add_pair_ignores_empty_input(window) -> None:
    before = set(window._instrument_boxes)

    window._new_symbol_edit.setText("   ")
    window._on_add_pair_clicked()

    assert set(window._instrument_boxes) == before


def test_provider_combo_includes_fake_and_switching_rebuilds_provider(window) -> None:
    assert window._provider_combo.findText("fake") >= 0

    window._provider_combo.setCurrentText("fake")

    assert window._provider.name == "fake"
    assert window._cfg.provider_name == "fake"


def test_strategy_param_specs_only_reference_real_strategies() -> None:
    assert set(_STRATEGY_PARAM_SPECS) <= set(STRATEGIES)


def test_update_chart_renders_a_candlestick_series(window) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [_make_candle(100.0 + i * 0.1, start + timedelta(hours=i)) for i in range(20)]

    window._update_chart(candles)

    series = window._chart.series()
    assert len(series) == 1
    assert series[0].count() == 20


def test_update_chart_replaces_previous_series_instead_of_stacking(window) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = [_make_candle(100.0, start + timedelta(hours=i)) for i in range(10)]
    second = [_make_candle(200.0, start + timedelta(hours=i)) for i in range(5)]

    window._update_chart(first)
    window._update_chart(second)

    series = window._chart.series()
    assert len(series) == 1
    assert series[0].count() == 5


class _FakeSession:
    def __init__(self, last_candles: dict[str, list[Candle]]) -> None:
        self.last_candles = last_candles


def test_last_candle_age_text_reports_seconds_minutes_hours(window) -> None:
    now = datetime.now(UTC)

    assert "waiting for first candle" in window._last_candle_age_text(_FakeSession({}))

    recent = _FakeSession({"FAKE": [_make_candle(1.0, now - timedelta(seconds=20))]})
    assert "20s ago" in window._last_candle_age_text(recent)

    minutes_old = _FakeSession({"FAKE": [_make_candle(1.0, now - timedelta(minutes=7))]})
    assert "7m ago" in window._last_candle_age_text(minutes_old)

    hours_old = _FakeSession({"FAKE": [_make_candle(1.0, now - timedelta(hours=3))]})
    assert "3.0h ago" in window._last_candle_age_text(hours_old)


def test_last_candle_age_text_picks_newest_across_symbols(window) -> None:
    now = datetime.now(UTC)
    session = _FakeSession(
        {
            "OLD": [_make_candle(1.0, now - timedelta(minutes=10), symbol="OLD")],
            "NEW": [_make_candle(1.0, now - timedelta(seconds=5), symbol="NEW")],
        }
    )

    assert "5s ago" in window._last_candle_age_text(session)
