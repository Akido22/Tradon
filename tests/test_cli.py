"""CLI integration tests — invoke real Typer commands end to end against
FakeProvider (see conftest.py) instead of mocking individual functions, so
these exercise the actual command wiring (option parsing, error messages,
table rendering) the same way a real `tradon ...` invocation would.
"""
from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from tradon.cli.main import app
from tradon.config import PROJECT_ROOT

runner = CliRunner()
SYMBOL = "FAKE"


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    yield
    db_path = PROJECT_ROOT / "data" / "tradon.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM signals WHERE symbol LIKE 'FAKE%'")
    conn.execute("DELETE FROM candles WHERE symbol LIKE 'FAKE%'")
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
async def _dispose_engine_after_test():
    # Same reasoning as test_runtime.py's identical fixture: the cached
    # engine is bound to whichever event loop created it, and each test
    # (CliRunner included — it runs the async command via asyncio.run())
    # gets its own loop.
    yield
    from tradon.db.session import get_engine, get_sessionmaker

    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def test_symbols_command_lists_provider_symbols(fake_provider) -> None:
    result = runner.invoke(app, ["symbols", "--provider", "fake"])

    assert result.exit_code == 0
    assert "FAKE" in result.output


def test_symbols_command_rejects_unknown_provider() -> None:
    result = runner.invoke(app, ["symbols", "--provider", "nope"])

    assert result.exit_code != 0
    assert "Unknown provider" in result.output


def test_run_once_renders_signals_table(fake_provider) -> None:
    result = runner.invoke(app, ["run", "--once", "--provider", "fake", "--symbols", SYMBOL])

    assert result.exit_code == 0, result.output
    assert "Signals" in result.output


def test_run_once_renders_startup_config_banner(fake_provider) -> None:
    result = runner.invoke(app, ["run", "--once", "--provider", "fake", "--symbols", SYMBOL])

    assert result.exit_code == 0, result.output
    assert "Monitoring configuration" in result.output
    assert "fake" in result.output
    assert SYMBOL in result.output
    # Whatever strategies are enabled in config.yaml should be named, not
    # left for the caller to go look up — sma_cross is enabled by default.
    assert "sma_cross" in result.output


def test_run_reports_clean_error_for_unsupported_symbol(fake_provider) -> None:
    result = runner.invoke(app, ["run", "--once", "--provider", "fake", "--symbols", "NOTREAL"])

    assert result.exit_code != 0
    assert "does not support" in result.output
    # No raw traceback leaked to the user — see the describe_error() fix.
    assert "Traceback" not in result.output


def test_backtest_single_combo_renders_metrics(fake_provider) -> None:
    result = runner.invoke(
        app,
        ["backtest", "--symbol", SYMBOL, "--strategy", "sma_cross", "--provider", "fake", "--limit", "300"],
    )

    assert result.exit_code == 0, result.output
    assert "Backtest" in result.output
    assert "Sharpe ratio" in result.output


def test_backtest_portfolio_mode_renders_combined_and_correlation(fake_provider) -> None:
    result = runner.invoke(
        app,
        [
            "backtest",
            "--symbol",
            "FAKE,FAKE2",
            "--strategy",
            "sma_cross,rsi",
            "--provider",
            "fake",
            "--limit",
            "300",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "per combo" in result.output
    assert "correlation" in result.output.lower()
    assert "shared capital" in result.output.lower()


def test_backtest_rejects_unknown_strategy(fake_provider) -> None:
    result = runner.invoke(
        app, ["backtest", "--symbol", SYMBOL, "--strategy", "not_a_strategy", "--provider", "fake"]
    )

    assert result.exit_code != 0
    assert "Unknown strategy" in result.output


def test_optimize_params_renders_ranked_table(fake_provider) -> None:
    result = runner.invoke(
        app,
        ["optimize-params", "--symbol", SYMBOL, "--strategy", "sma_cross", "--provider", "fake", "--limit", "300"],
    )

    assert result.exit_code == 0, result.output
    assert "Optimize" in result.output


def test_walk_forward_renders_out_of_sample_summary(fake_provider) -> None:
    result = runner.invoke(
        app,
        [
            "walk-forward",
            "--symbol",
            SYMBOL,
            "--strategy",
            "sma_cross",
            "--provider",
            "fake",
            "--limit",
            "300",
            "--splits",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Walk-forward" in result.output
    assert "Out-of-sample" in result.output


def test_walk_forward_rejects_too_few_splits(fake_provider) -> None:
    result = runner.invoke(
        app, ["walk-forward", "--symbol", SYMBOL, "--strategy", "sma_cross", "--provider", "fake", "--splits", "1"]
    )

    assert result.exit_code != 0
    assert "at least 2" in result.output


def test_train_model_requires_ml_extra_or_runs(fake_provider, tmp_path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    model_path = str(tmp_path / "model.joblib")

    result = runner.invoke(
        app,
        [
            "train-model",
            "--symbol",
            SYMBOL,
            "--provider",
            "fake",
            "--limit",
            "300",
            "--horizon",
            "5",
            "--model-path",
            model_path,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ML model" in result.output
    assert "Feature importance" in result.output
    assert (tmp_path / "model.joblib").exists()


def test_ml_walk_forward_runs_without_crashing(fake_provider) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")

    result = runner.invoke(
        app,
        [
            "ml-walk-forward",
            "--symbol",
            SYMBOL,
            "--provider",
            "fake",
            "--limit",
            "300",
            "--splits",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    # FakeProvider only ever returns ~300 candles regardless of --limit, so
    # whether that's enough for any fold to actually train is incidental —
    # what matters here is the command wires up and renders either outcome
    # cleanly, not a raw traceback.
    assert "ML walk-forward" in result.output or "No fold had enough data" in result.output


def test_ml_walk_forward_rejects_too_few_splits(fake_provider) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")

    result = runner.invoke(
        app, ["ml-walk-forward", "--symbol", SYMBOL, "--provider", "fake", "--splits", "1"]
    )

    assert result.exit_code != 0
    assert "at least 2" in result.output
