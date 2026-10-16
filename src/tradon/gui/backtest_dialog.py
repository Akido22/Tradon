"""Backtest dialog — same rule as the rest of the GUI: reuses tradon.backtest
directly (see ARCHITECTURE.md §6), no separate analysis logic, no orders.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from tradon.backtest import run_backtest
from tradon.data.base import DataProvider
from tradon.runtime import STRATEGIES, describe_error


class BacktestDialog(QDialog):
    def __init__(
        self,
        parent,
        provider: DataProvider,
        instruments: list[str],
        timeframe: str,
        strategy_params: dict[str, dict],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Backtest")
        self.resize(420, 420)
        self._provider = provider
        self._timeframe = timeframe
        self._strategy_params = strategy_params

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._symbol_combo = QComboBox()
        self._symbol_combo.addItems(instruments)
        form.addRow("Symbol", self._symbol_combo)

        self._strategy_combo = QComboBox()
        self._strategy_combo.addItems(list(STRATEGIES))
        form.addRow("Strategy", self._strategy_combo)

        self._limit_spin = QSpinBox()
        self._limit_spin.setRange(50, 5000)
        self._limit_spin.setValue(500)
        form.addRow("Candles", self._limit_spin)

        self._balance_spin = QDoubleSpinBox()
        self._balance_spin.setRange(100, 10_000_000)
        self._balance_spin.setValue(10000)
        self._balance_spin.setDecimals(0)
        form.addRow("Balance", self._balance_spin)
        layout.addLayout(form)

        run_row = QHBoxLayout()
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self._run_button)
        self._status_label = QLabel("")
        run_row.addWidget(self._status_label)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self._results = QTableWidget(0, 2)
        self._results.setHorizontalHeaderLabels(["Metric", "Value"])
        self._results.verticalHeader().setVisible(False)
        layout.addWidget(self._results)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        layout.addWidget(close_button)

    def _on_run_clicked(self) -> None:
        asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        self._run_button.setEnabled(False)
        self._status_label.setText("Fetching candles…")
        symbol = self._symbol_combo.currentText()
        strategy_name = self._strategy_combo.currentText()
        limit = self._limit_spin.value()
        balance = Decimal(str(self._balance_spin.value()))

        try:
            candles = await self._provider.get_ohlcv(symbol, self._timeframe, limit=limit)
        except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
            self._status_label.setText(f"Error: {describe_error(exc)}")
            self._run_button.setEnabled(True)
            return

        if not candles:
            self._status_label.setText("No candles returned.")
            self._run_button.setEnabled(True)
            return

        try:
            strategy = STRATEGIES[strategy_name](self._strategy_params.get(strategy_name, {}))
        except ValueError as exc:  # e.g. the ml strategy without a trained model_path
            self._status_label.setText(f"Error: {exc}")
            self._run_button.setEnabled(True)
            return
        result = run_backtest(candles, strategy, starting_balance=balance)
        self._status_label.setText(f"{len(candles)} candles, {result.total_trades} trades")
        self._populate_results(result)
        self._run_button.setEnabled(True)

    def _populate_results(self, result) -> None:
        rows = [
            ("Starting balance", f"{result.starting_balance:.2f}"),
            ("Ending balance", f"{result.ending_balance:.2f}"),
            ("Total PnL", f"{result.total_pnl:+.2f}"),
            ("Win rate", f"{result.win_rate:.1%}" if result.trades else "n/a"),
            ("Profit factor", f"{result.profit_factor:.2f}" if result.profit_factor is not None else "n/a"),
            ("Max drawdown", f"{result.max_drawdown:.1%}"),
            ("Wins / Losses", f"{result.wins} / {result.losses}"),
        ]
        self._results.setRowCount(len(rows))
        for row, (metric, value) in enumerate(rows):
            self._results.setItem(row, 0, QTableWidgetItem(metric))
            self._results.setItem(row, 1, QTableWidgetItem(value))
