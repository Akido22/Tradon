"""Web dashboard endpoint tests — a real FastAPI TestClient against the
real app (including its lifespan), with load_runtime_config()/build_provider
monkeypatched so the lifespan's LiveSession never makes a network call: an
empty instrument list means LiveSession is constructed but never started.
Everything else (routing, auth, DB access) runs for real.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from tradon.config import PROJECT_ROOT
from tradon.db.models import User
from tradon.db.session import get_sessionmaker
from tradon.runtime import RuntimeConfig
from tradon.web.auth import hash_password

TEST_EMAIL = "pytest-web-test@example.invalid"
TEST_PASSWORD = "correct horse battery staple"


def _test_runtime_cfg(provider_name: str = "fake") -> RuntimeConfig:
    return RuntimeConfig(
        provider_name=provider_name,
        timeframe="1h",
        instruments=[],  # empty on purpose — lifespan constructs but never starts LiveSession
        strategy_specs=[{"name": "sma_cross", "enabled": True, "params": {}}],
        poll_interval_seconds=60,
    )


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    yield
    db_path = PROJECT_ROOT / "data" / "tradon.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM users WHERE email = ?", (TEST_EMAIL,))
    conn.execute("DELETE FROM candles WHERE symbol LIKE 'FAKE%'")
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
async def _dispose_engine_after_test():
    yield
    from tradon.db.session import get_engine, get_sessionmaker as _get_sessionmaker

    await get_engine().dispose()
    get_engine.cache_clear()
    _get_sessionmaker.cache_clear()


@pytest.fixture
async def test_user():
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        session.add(User(email=TEST_EMAIL, hashed_password=hash_password(TEST_PASSWORD), is_active=True))
        await session.commit()
    return TEST_EMAIL


@pytest.fixture
def client(fake_provider, monkeypatch):
    import tradon.web.app as web_app

    monkeypatch.setattr(web_app, "load_runtime_config", _test_runtime_cfg)
    with TestClient(web_app.app) as test_client:
        yield test_client


def _login(client: TestClient) -> None:
    response = client.post(
        "/login", data={"email": TEST_EMAIL, "password": TEST_PASSWORD}, follow_redirects=False
    )
    assert response.status_code == 303, response.text


def test_login_page_renders(client: TestClient) -> None:
    response = client.get("/login")

    assert response.status_code == 200
    assert "Login" in response.text or "login" in response.text.lower()


def test_dashboard_redirects_to_login_when_unauthenticated(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/login"


def test_login_with_wrong_password_returns_401(client: TestClient, test_user) -> None:
    response = client.post("/login", data={"email": TEST_EMAIL, "password": "wrong"}, follow_redirects=False)

    assert response.status_code == 401
    assert "tradon_session" not in response.cookies


def test_login_with_correct_password_sets_cookie_and_redirects(client: TestClient, test_user) -> None:
    response = client.post(
        "/login", data={"email": TEST_EMAIL, "password": TEST_PASSWORD}, follow_redirects=False
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "tradon_session" in response.cookies


def test_dashboard_loads_after_login(client: TestClient, test_user) -> None:
    _login(client)

    response = client.get("/")

    assert response.status_code == 200
    assert TEST_EMAIL in response.text


def test_logout_clears_session(client: TestClient, test_user) -> None:
    _login(client)
    client.post("/logout")

    response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/login"


def test_api_status_requires_auth(client: TestClient) -> None:
    response = client.get("/api/status")

    assert response.status_code == 401


def test_api_status_returns_mode_when_authenticated(client: TestClient, test_user) -> None:
    _login(client)

    response = client.get("/api/status")

    assert response.status_code == 200
    assert "mode" in response.json()


def test_api_candles_requires_auth(client: TestClient) -> None:
    response = client.get("/api/candles/FAKE")

    assert response.status_code == 401


def test_api_signals_requires_auth(client: TestClient) -> None:
    response = client.get("/api/signals")

    assert response.status_code == 401


def test_portfolio_page_requires_auth(client: TestClient) -> None:
    response = client.get("/portfolio", follow_redirects=False)

    assert response.status_code in (302, 307)


def test_portfolio_backtest_requires_auth(client: TestClient) -> None:
    response = client.post("/api/portfolio-backtest", json={"symbols": ["FAKE"], "strategies": ["sma_cross"]})

    assert response.status_code == 401


def test_portfolio_backtest_rejects_unknown_strategy(client: TestClient, test_user) -> None:
    _login(client)

    response = client.post(
        "/api/portfolio-backtest", json={"symbols": ["FAKE"], "strategies": ["not_a_strategy"]}
    )

    assert response.status_code == 400
    assert "Unknown strategy" in response.json()["detail"]


def test_portfolio_backtest_runs_against_fake_provider(client: TestClient, test_user, monkeypatch) -> None:
    import tradon.web.app as web_app
    from tests.conftest import FakeProvider

    monkeypatch.setattr(web_app, "build_provider", lambda cfg: FakeProvider())
    _login(client)

    response = client.post(
        "/api/portfolio-backtest",
        json={"symbols": ["FAKE", "FAKE2"], "strategies": ["sma_cross", "rsi"], "limit": 300, "balance": 10000},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "combined" in body
    assert "correlation" in body
    assert len(body["per_combo"]) == 4  # 2 symbols x 2 strategies
    assert body["correlation"]["FAKE"]["FAKE"] == 1.0


def test_portfolio_backtest_validates_weights_sum(client: TestClient, test_user, monkeypatch) -> None:
    import tradon.web.app as web_app
    from tests.conftest import FakeProvider

    monkeypatch.setattr(web_app, "build_provider", lambda cfg: FakeProvider())
    _login(client)

    response = client.post(
        "/api/portfolio-backtest",
        json={
            "symbols": ["FAKE"],
            "strategies": ["sma_cross"],
            "limit": 300,
            "weights": {"FAKE/sma_cross": 0.5},
        },
    )

    assert response.status_code == 400
    assert "sum to 1" in response.json()["detail"]
