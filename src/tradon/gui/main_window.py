"""GUI — Presentation Layer (see ARCHITECTURE.md §3.7).

Same rule as the CLI: no analysis logic here. This window only toggles which
instruments/strategies are active, calls tradon.runtime.poll_once(), and
renders whatever comes back. qasync merges the asyncio and Qt event loops
onto one thread, so the polling loop is just an asyncio.Task — no threads,
no cross-thread signal plumbing needed.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from PySide6.QtCharts import QBarCategoryAxis, QCandlestickSeries, QCandlestickSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy import select

from tradon.data.base import DataProvider
from tradon.db.models import SignalRecord, StrategyConfig
from tradon.db.session import get_engine, get_sessionmaker
from tradon.gui.backtest_dialog import BacktestDialog
from tradon.runtime import (
    PROVIDERS,
    STRATEGIES,
    LiveSession,
    RuntimeConfig,
    build_provider,
    build_risk_manager,
    build_signal_engine,
    describe_error,
    load_runtime_config,
    normalize_symbol,
)
from tradon.strategy.base import SignalDirection, Strategy

_DIRECTION_COLOR = {
    SignalDirection.LONG: QColor("#1D9E75"),
    SignalDirection.SHORT: QColor("#D85A30"),
}

# (key, min, max, default) per strategy — drives the param-editing dialog.
_STRATEGY_PARAM_SPECS: dict[str, list[tuple[str, int, int, int]]] = {
    "sma_cross": [("fast_period", 2, 200, 10), ("slow_period", 2, 500, 30)],
    "rsi": [("period", 2, 100, 14), ("oversold", 1, 49, 30), ("overbought", 51, 99, 70)],
    "macd": [("fast_period", 2, 100, 12), ("slow_period", 3, 200, 26), ("signal_period", 2, 50, 9)],
    # std_dev is fractional (e.g. 2.0) — not editable via this int-only spin dialog, only `period` is.
    "bollinger": [("period", 5, 100, 20)],
    # atr_multiplier is fractional too — same story, only the two periods are editable here.
    "atr_breakout": [("channel_period", 5, 200, 20), ("atr_period", 2, 100, 14)],
    "ichimoku": [
        ("tenkan_period", 2, 100, 9),
        ("kijun_period", 3, 200, 26),
        ("senkou_b_period", 4, 300, 52),
        ("displacement", 1, 200, 26),
    ],
}


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Tradon")
        self.resize(900, 560)

        self._cfg: RuntimeConfig = load_runtime_config()
        self._provider: DataProvider = build_provider(self._cfg)
        self._poll_session: LiveSession | None = None
        self._instrument_boxes: dict[str, QCheckBox] = {}
        self._strategy_boxes: dict[str, QCheckBox] = {}
        self._strategy_params: dict[str, dict] = self._initial_strategy_params()
        # Overridable so tests can point this at an isolated scope instead
        # of this machine's real saved GUI preferences (checked
        # instruments, provider choice, strategy params) — reading or,
        # worse, clearing the real ones as a side effect of running the
        # test suite would be a bug in the tests, not a thing to work around.
        self._settings = settings if settings is not None else QSettings("Tradon", "Tradon")

        self._build_ui()
        self._restore_settings()
        self._refresh_status("Idle")
        asyncio.ensure_future(self._load_history())

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)

        root_layout.addWidget(self._build_sidebar(), 0)
        root_layout.addWidget(self._build_main_area(), 1)

        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        self._instruments_layout = layout

        layout.addWidget(QLabel("Instruments"))
        for symbol in self._cfg.instruments:
            box = QCheckBox(symbol)
            box.setChecked(True)
            self._instrument_boxes[symbol] = box
            layout.addWidget(box)

        add_row = QHBoxLayout()
        self._new_symbol_edit = QLineEdit()
        self._new_symbol_edit.setPlaceholderText("e.g. USDCHF")
        add_button = QPushButton("Add pair")
        add_button.clicked.connect(self._on_add_pair_clicked)
        add_row.addWidget(self._new_symbol_edit)
        add_row.addWidget(add_button)
        self._add_row_index = layout.count()
        layout.addLayout(add_row)

        layout.addSpacing(16)
        layout.addWidget(QLabel("Strategies"))
        enabled_by_name = {s["name"]: s.get("enabled", True) for s in self._cfg.strategy_specs}
        for name in STRATEGIES:
            row = QHBoxLayout()
            box = QCheckBox(name)
            box.setChecked(enabled_by_name.get(name, True))
            self._strategy_boxes[name] = box
            row.addWidget(box)
            if name in _STRATEGY_PARAM_SPECS:
                row.addStretch(1)
                gear = QToolButton()
                gear.setText("...")
                gear.setToolTip(f"Edit {name} parameters")
                gear.clicked.connect(lambda _checked=False, n=name: self._on_edit_params_clicked(n))
                row.addWidget(gear)
            layout.addLayout(row)

        layout.addStretch(1)
        return sidebar

    def _build_main_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Source"))
        self._provider_combo = QComboBox()
        self._provider_combo.addItems(list(PROVIDERS))
        self._provider_combo.setCurrentText(self._cfg.provider_name)
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        toolbar.addWidget(self._provider_combo)
        self._start_stop_button = QPushButton("Start")
        self._start_stop_button.clicked.connect(self._on_start_stop_clicked)
        toolbar.addWidget(self._start_stop_button)
        toolbar.addWidget(QLabel("Poll every (s)"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(5, 3600)
        self._interval_spin.setValue(self._cfg.poll_interval_seconds)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        toolbar.addWidget(self._interval_spin)
        toolbar.addWidget(QLabel("Chart"))
        self._chart_symbol_combo = QComboBox()
        self._chart_symbol_combo.currentTextChanged.connect(self._on_chart_symbol_changed)
        toolbar.addWidget(self._chart_symbol_combo)
        backtest_button = QPushButton("Backtest")
        backtest_button.clicked.connect(self._on_backtest_clicked)
        toolbar.addWidget(backtest_button)
        toolbar.addStretch(1)
        self._connection_label = QLabel(f"{self._provider.name}: idle")
        toolbar.addWidget(self._connection_label)
        # A pulsing dot plus wall-clock time proves the app's event loop is
        # actually ticking every second (not frozen) even when nothing else
        # on screen changes — "last poll" only updates on a signal, which on
        # a quiet instrument/timeframe can be silent for a very long time
        # and looks indistinguishable from a hung app.
        self._heartbeat_label = QLabel("")
        toolbar.addWidget(self._heartbeat_label)
        layout.addLayout(toolbar)

        self._chart = QChart()
        self._chart.legend().hide()
        self._chart_view = QChartView(self._chart)
        self._chart_view.setFixedHeight(220)
        layout.addWidget(self._chart_view)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Strategy", "Direction", "Size", "Stop", "Target", "Reason"]
        )
        self._table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table)

        return area

    def _initial_strategy_params(self) -> dict[str, dict]:
        from_config = {s["name"]: s.get("params", {}) for s in self._cfg.strategy_specs}
        params = {}
        for name, spec in _STRATEGY_PARAM_SPECS.items():
            defaults = {key: default for key, _min, _max, default in spec}
            params[name] = {**defaults, **from_config.get(name, {})}
        return params

    def _on_edit_params_clicked(self, name: str) -> None:
        spec = _STRATEGY_PARAM_SPECS[name]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{name} parameters")
        form = QFormLayout(dialog)

        spinboxes: dict[str, QSpinBox] = {}
        for key, minimum, maximum, _default in spec:
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(self._strategy_params[name].get(key, minimum))
            form.addRow(key, spin)
            spinboxes[key] = spin

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.Accepted:
            self._strategy_params[name] = {key: spin.value() for key, spin in spinboxes.items()}

    def _on_backtest_clicked(self) -> None:
        instruments = list(self._instrument_boxes)
        if not instruments:
            QMessageBox.warning(self, "Tradon", "Add an instrument first.")
            return
        dialog = BacktestDialog(self, self._provider, instruments, self._cfg.timeframe, self._strategy_params)
        dialog.show()

    # -- History -------------------------------------------------------------

    async def _load_history(self, limit: int = 50) -> None:
        session_maker = get_sessionmaker()
        async with session_maker() as session:
            result = await session.execute(
                select(SignalRecord, StrategyConfig.name)
                .join(StrategyConfig, SignalRecord.strategy_id == StrategyConfig.id)
                .order_by(SignalRecord.id.desc())
                .limit(limit)
            )
            rows = result.all()

        for sig, strategy_name in reversed(rows):
            self._add_history_row(sig, strategy_name)
        if rows:
            self._refresh_status(f"Loaded {len(rows)} signals from history")

    def _add_history_row(self, sig, strategy_name: str) -> None:
        direction = SignalDirection(sig.direction)
        self._table.insertRow(0)
        values = [
            sig.timestamp.strftime("%Y-%m-%d %H:%M"),
            sig.symbol,
            strategy_name,
            direction.value.capitalize(),
            f"{sig.size:.4f}" if sig.size is not None else "n/a",
            f"{sig.stop_loss:.5f}" if sig.stop_loss is not None else "n/a",
            f"{sig.take_profit:.5f}" if sig.take_profit is not None else "n/a",
            sig.reason or "",
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col == 3:
                item.setForeground(_DIRECTION_COLOR[direction])
            self._table.setItem(0, col, item)

    # -- Settings persistence --------------------------------------------------

    def _restore_settings(self) -> None:
        s = self._settings

        provider = s.value("provider", "")
        if provider and provider in PROVIDERS:
            self._provider_combo.setCurrentText(provider)

        try:
            extra_instruments = json.loads(s.value("extra_instruments", "[]"))
        except (TypeError, ValueError):
            extra_instruments = []
        for symbol in extra_instruments:
            self._add_instrument_checkbox(symbol)

        checked_raw = s.value("checked_instruments", None)
        if checked_raw is not None:
            try:
                checked = set(json.loads(checked_raw))
                for symbol, box in self._instrument_boxes.items():
                    box.setChecked(symbol in checked)
            except (TypeError, ValueError):
                pass

        checked_strategies_raw = s.value("checked_strategies", None)
        if checked_strategies_raw is not None:
            try:
                checked = set(json.loads(checked_strategies_raw))
                for name, box in self._strategy_boxes.items():
                    box.setChecked(name in checked)
            except (TypeError, ValueError):
                pass

        interval = s.value("poll_interval", None)
        if interval:
            try:
                self._interval_spin.setValue(int(interval))
            except (TypeError, ValueError):
                pass

        for name in list(self._strategy_params):
            saved = s.value(f"params/{name}", None)
            if not saved:
                continue
            try:
                self._strategy_params[name].update(json.loads(saved))
            except (TypeError, ValueError):
                pass

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("provider", self._provider_combo.currentText())
        s.setValue("poll_interval", self._interval_spin.value())
        s.setValue(
            "checked_instruments",
            json.dumps([sym for sym, box in self._instrument_boxes.items() if box.isChecked()]),
        )
        s.setValue(
            "extra_instruments",
            json.dumps([sym for sym in self._instrument_boxes if sym not in self._cfg.instruments]),
        )
        s.setValue(
            "checked_strategies",
            json.dumps([name for name, box in self._strategy_boxes.items() if box.isChecked()]),
        )
        for name, params in self._strategy_params.items():
            s.setValue(f"params/{name}", json.dumps(params))

    # -- Start/stop --------------------------------------------------------

    def _on_provider_changed(self, name: str) -> None:
        self._cfg.provider_name = name
        self._provider = build_provider(self._cfg)
        self._connection_label.setText(f"{self._provider.name}: idle")

    def _on_add_pair_clicked(self) -> None:
        symbol = normalize_symbol(self._cfg.provider_name, self._new_symbol_edit.text())
        if not symbol:
            return
        self._add_instrument_checkbox(symbol)
        self._new_symbol_edit.clear()

    def _add_instrument_checkbox(self, symbol: str, checked: bool = True) -> None:
        if symbol in self._instrument_boxes:
            self._instrument_boxes[symbol].setChecked(checked)
            return
        box = QCheckBox(symbol)
        box.setChecked(checked)
        self._instrument_boxes[symbol] = box
        self._instruments_layout.insertWidget(self._add_row_index, box)
        self._add_row_index += 1

    def _on_start_stop_clicked(self) -> None:
        if self._poll_session is None:
            asyncio.ensure_future(self._start_monitoring())
        else:
            self._stop_monitoring()

    def _on_interval_changed(self, value: int) -> None:
        if self._poll_session is not None:
            self._poll_session.reschedule(value)

    async def _start_monitoring(self) -> None:
        instruments = [s for s, box in self._instrument_boxes.items() if box.isChecked()]
        strategy_names = [n for n, box in self._strategy_boxes.items() if box.isChecked()]
        if not instruments or not strategy_names:
            QMessageBox.warning(self, "Tradon", "Pick at least one instrument and one strategy.")
            return

        self._connection_label.setText(f"{self._provider.name}: checking symbols...")
        try:
            unknown = [s for s in instruments if not await self._provider.supports_symbol(s)]
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            QMessageBox.critical(self, "Tradon", f"Could not reach {self._provider.name}:\n{describe_error(exc)}")
            self._connection_label.setText(f"{self._provider.name}: error")
            return

        if unknown:
            QMessageBox.warning(self, "Tradon", f"{self._provider.name} does not support: {', '.join(unknown)}")
            return

        try:
            strategies: list[Strategy] = [
                STRATEGIES[name](self._strategy_params.get(name, {})) for name in strategy_names
            ]
        except ValueError as exc:  # e.g. the ml strategy without a trained model_path
            QMessageBox.critical(self, "Tradon", f"Could not start strategy:\n{exc}")
            self._connection_label.setText(f"{self._provider.name}: error")
            return
        signal_engine = build_signal_engine(self._cfg)
        risk_manager = build_risk_manager(self._cfg)

        self._chart_symbol_combo.blockSignals(True)
        self._chart_symbol_combo.clear()
        self._chart_symbol_combo.addItems(instruments)
        self._chart_symbol_combo.blockSignals(False)

        self._start_stop_button.setText("Stop")
        self._provider_combo.setEnabled(False)
        self._connection_label.setText(f"{self._provider.name}: connecting...")
        self._poll_session = LiveSession(
            self._provider, instruments, self._cfg.timeframe, strategies, signal_engine,
            risk_manager, self._cfg.risk_account_balance,
            on_signals=self._on_signals,
            on_error=self._on_poll_error,
        )
        self._poll_session.start(self._interval_spin.value())
        asyncio.ensure_future(self._watch_mode(self._poll_session))

    async def _watch_mode(self, session: LiveSession) -> None:
        last_mode = None
        pulse = False
        while self._poll_session is session:
            if session.mode != last_mode:
                last_mode = session.mode
                label = {"starting": "connecting...", "realtime": "live (realtime)", "poll": "live (polling)"}
                self._connection_label.setText(f"{self._provider.name}: {label.get(session.mode, session.mode)}")
            # Candles land in session.last_candles as soon as warm-up finishes
            # and on every new bar after that, but on_signals only fires when
            # a strategy actually produces a signal — without this, the chart
            # would stay blank until the first signal, which can be a long
            # wait (or never) on a quiet instrument.
            self._refresh_chart()
            pulse = not pulse
            self._heartbeat_label.setText(f"{'●' if pulse else '○'} {self._last_candle_age_text(session)}")
            await asyncio.sleep(1)

    def _last_candle_age_text(self, session: LiveSession) -> str:
        """How long ago the most recently received candle (across all
        instruments) actually arrived — proof data is flowing, not just
        that the window is open. Separate from the pulse: the pulse shows
        the event loop is alive even when nothing else has happened yet
        (e.g. still warming up, or waiting for an hourly bar to close)."""
        newest = None
        for candles in session.last_candles.values():
            if candles and (newest is None or candles[-1].timestamp > newest):
                newest = candles[-1].timestamp
        if newest is None:
            return "waiting for first candle..."
        age_seconds = (datetime.now(UTC) - newest).total_seconds()
        if age_seconds < 60:
            return f"last candle {int(age_seconds)}s ago"
        if age_seconds < 3600:
            return f"last candle {int(age_seconds // 60)}m ago"
        return f"last candle {age_seconds / 3600:.1f}h ago"

    def _stop_monitoring(self) -> None:
        if self._poll_session is not None:
            self._poll_session.stop()
            self._poll_session = None
        self._start_stop_button.setText("Start")
        self._provider_combo.setEnabled(True)
        self._connection_label.setText(f"{self._provider.name}: idle")
        self._heartbeat_label.setText("")
        self._refresh_status("Stopped")

    def _on_signals(self, signals) -> None:
        for signal in signals:
            self._prepend_row(signal)
        self._refresh_status(f"Last poll {datetime.now():%H:%M:%S}")
        self._refresh_chart()

    def _on_poll_error(self, exc: Exception) -> None:
        QMessageBox.critical(self, "Tradon", f"Polling stopped:\n{describe_error(exc)}")
        self._stop_monitoring()

    def _on_chart_symbol_changed(self, _symbol: str) -> None:
        self._refresh_chart()

    def _refresh_chart(self) -> None:
        symbol = self._chart_symbol_combo.currentText()
        if not symbol or self._poll_session is None:
            return
        candles = self._poll_session.last_candles.get(symbol)
        if not candles:
            return
        self._update_chart(candles)

    def _update_chart(self, candles: list) -> None:
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)

        window = candles[-100:]
        series = QCandlestickSeries()
        series.setIncreasingColor(QColor("#1D9E75"))
        series.setDecreasingColor(QColor("#D85A30"))
        for i, c in enumerate(window):
            series.append(QCandlestickSet(float(c.open), float(c.high), float(c.low), float(c.close), i))
        self._chart.addSeries(series)

        # QCandlestickSeries needs an explicit category X axis — without one
        # attached, Qt Charts renders only the Y-axis grid and no candles.
        axis_x = QBarCategoryAxis()
        axis_x.append([c.timestamp.strftime("%m-%d %H:%M") for c in window])
        self._chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        self._chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

    # -- Rendering helpers -------------------------------------------------

    def _prepend_row(self, signal_with_plan) -> None:
        signal, plan = signal_with_plan.signal, signal_with_plan.plan
        self._table.insertRow(0)
        values = [
            datetime.now().strftime("%H:%M:%S"),
            signal.symbol,
            signal.strategy,
            signal.direction.value.capitalize(),
            f"{plan.size:.4f}" if plan else "n/a",
            f"{plan.stop_loss:.5f}" if plan else "n/a",
            f"{plan.take_profit:.5f}" if plan else "n/a",
            signal.reason,
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col == 3:
                item.setForeground(_DIRECTION_COLOR[signal.direction])
            self._table.setItem(0, col, item)

    def _refresh_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    # -- Shutdown ------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        self._save_settings()
        session = self._poll_session
        if session is not None:
            session.stop()
        asyncio.ensure_future(self._shutdown(session))
        super().closeEvent(event)

    @staticmethod
    async def _shutdown(session: LiveSession | None) -> None:
        # Qt's closeEvent can't be awaited, so this runs as a background
        # task — but it still matters: without waiting for the cancelled
        # session to actually unwind first, dispose() can race a
        # connection the task hasn't released yet.
        if session is not None:
            await session.wait_closed()
        await get_engine().dispose()
