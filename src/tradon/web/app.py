"""Web dashboard — Presentation Layer (see ARCHITECTURE.md §3.7, Этап 4).

FastAPI + Jinja2 templates ("проще для старта" per §4) — no separate frontend
build. Same rule as CLI/GUI/Telegram: no analysis logic here, it runs its
own LiveSession against tradon.runtime and renders/broadcasts what comes
back. Session-cookie auth against the `users` table (create accounts with
`tradon create-user` — see ARCHITECTURE.md §2 Этап 4 "многопользовательский
доступ").
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel
from sqlalchemy import select

from tradon.backtest import BacktestResult, combo_key, correlation_matrix, run_portfolio
from tradon.config import get_settings
from tradon.db.models import Candle as CandleRow
from tradon.db.models import SignalRecord, StrategyConfig
from tradon.db.session import get_engine, get_sessionmaker
from tradon.runtime import (
    STRATEGIES,
    LiveSession,
    RuntimeConfig,
    SignalWithPlan,
    build_provider,
    build_risk_manager,
    build_signal_engine,
    build_strategies,
    describe_error,
    load_runtime_config,
    normalize_symbol,
)
from tradon.web.auth import authenticate

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
SESSION_COOKIE = "tradon_session"


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - a dead socket shouldn't break the others
                self.active.discard(ws)


manager = ConnectionManager()
poll_session: LiveSession | None = None
runtime_cfg: RuntimeConfig | None = None


def _signal_to_json(sp: SignalWithPlan) -> dict:
    s, plan = sp.signal, sp.plan
    return {
        "type": "signal",
        "symbol": s.symbol,
        "strategy": s.strategy,
        "direction": s.direction.value,
        "strength": float(s.strength),
        "reason": s.reason,
        "timestamp": s.timestamp.isoformat(),
        "size": float(plan.size) if plan else None,
        "stop_loss": float(plan.stop_loss) if plan else None,
        "take_profit": float(plan.take_profit) if plan else None,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global poll_session, runtime_cfg
    runtime_cfg = load_runtime_config()
    provider = build_provider(runtime_cfg)
    strategies = build_strategies(runtime_cfg)
    signal_engine = build_signal_engine(runtime_cfg)
    risk_manager = build_risk_manager(runtime_cfg)

    async def on_signals(signals: list[SignalWithPlan]) -> None:
        for sp in signals:
            await manager.broadcast(_signal_to_json(sp))

    async def on_error(exc: Exception) -> None:
        await manager.broadcast({"type": "error", "message": str(exc)})

    poll_session = LiveSession(
        provider, runtime_cfg.instruments, runtime_cfg.timeframe, strategies, signal_engine,
        risk_manager, runtime_cfg.risk_account_balance,
        on_signals=on_signals, on_error=on_error,
    )
    if runtime_cfg.instruments:
        poll_session.start(runtime_cfg.poll_interval_seconds)

    yield

    poll_session.stop()
    await poll_session.wait_closed()
    await get_engine().dispose()


app = FastAPI(lifespan=lifespan)


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_settings().web_secret_key)


def _session_email(conn) -> str | None:
    token = conn.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = _serializer().loads(token)
    except BadSignature:
        return None
    return data.get("email")


async def current_user(request: Request) -> str | None:
    return _session_email(request)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        user = await authenticate(session, email, password)
    if user is None:
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid email or password"}, status_code=401
        )
    token = _serializer().dumps({"email": user.email})
    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
    return response


@app.post("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user: str | None = Depends(current_user)):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "provider": runtime_cfg.provider_name,
            "instruments": runtime_cfg.instruments,
            "timeframe": runtime_cfg.timeframe,
            "poll_interval": runtime_cfg.poll_interval_seconds,
        },
    )


@app.get("/api/status")
async def api_status(user: str | None = Depends(current_user)):
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    return {"mode": poll_session.mode if poll_session is not None else "idle"}


@app.get("/api/signals")
async def api_signals(limit: int = 50, user: str | None = Depends(current_user)):
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        result = await session.execute(
            select(SignalRecord, StrategyConfig.name)
            .join(StrategyConfig, SignalRecord.strategy_id == StrategyConfig.id)
            .order_by(SignalRecord.id.desc())
            .limit(limit)
        )
        rows = result.all()
    return [
        {
            "symbol": sig.symbol,
            "strategy": strategy_name,
            "direction": sig.direction,
            "strength": float(sig.strength),
            "reason": sig.reason,
            "timestamp": sig.timestamp.isoformat(),
            "size": float(sig.size) if sig.size is not None else None,
            "stop_loss": float(sig.stop_loss) if sig.stop_loss is not None else None,
            "take_profit": float(sig.take_profit) if sig.take_profit is not None else None,
        }
        for sig, strategy_name in rows
    ]


@app.get("/api/candles/{symbol}")
async def api_candles(symbol: str, limit: int = 200, user: str | None = Depends(current_user)):
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    symbol = normalize_symbol(runtime_cfg.provider_name, symbol)
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        result = await session.execute(
            select(CandleRow)
            .where(CandleRow.symbol == symbol, CandleRow.timeframe == runtime_cfg.timeframe)
            .order_by(CandleRow.timestamp.desc())
            .limit(limit)
        )
        rows = list(reversed(result.scalars().all()))
    return [
        {
            "time": int(c.timestamp.timestamp()),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
        }
        for c in rows
    ]


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    if _session_email(websocket) is None:
        await websocket.close(code=1008)
        return
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class PortfolioRequest(BaseModel):
    symbols: list[str]
    strategies: list[str]
    timeframe: str | None = None
    limit: int = 500
    balance: float = 10000
    # Capital share per "symbol/strategy" combo for the shared-account
    # `combined` run (see simulate_portfolio() in backtest.py). Omit for an
    # equal split across combos.
    weights: dict[str, float] | None = None


def _result_to_json(result: BacktestResult) -> dict:
    return {
        "starting_balance": float(result.starting_balance),
        "ending_balance": float(result.ending_balance),
        "total_pnl": float(result.total_pnl),
        "win_rate": float(result.win_rate) if result.trades else None,
        "profit_factor": float(result.profit_factor) if result.profit_factor is not None else None,
        "sharpe_ratio": float(result.sharpe_ratio) if result.sharpe_ratio is not None else None,
        "sortino_ratio": float(result.sortino_ratio) if result.sortino_ratio is not None else None,
        "max_drawdown": float(result.max_drawdown),
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "equity_curve": [{"time": int(t.timestamp()), "equity": float(e)} for t, e in result.equity_curve],
    }


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request, user: str | None = Depends(current_user)):
    if user is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse(
        request,
        "portfolio.html",
        {
            "user": user,
            "instruments": runtime_cfg.instruments,
            "strategies": list(STRATEGIES),
            "timeframe": runtime_cfg.timeframe,
        },
    )


@app.post("/api/portfolio-backtest")
async def api_portfolio_backtest(req: PortfolioRequest, user: str | None = Depends(current_user)):
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    if not req.symbols or not req.strategies:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Pick at least one symbol and one strategy")
    unknown_strategies = [s for s in req.strategies if s not in STRATEGIES]
    if unknown_strategies:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown strategy: {', '.join(unknown_strategies)}")

    provider = build_provider(runtime_cfg)
    timeframe = req.timeframe or runtime_cfg.timeframe
    params_by_strategy = {s["name"]: s.get("params", {}) for s in runtime_cfg.strategy_specs}

    symbols = [normalize_symbol(runtime_cfg.provider_name, s) for s in req.symbols]

    candles_by_symbol: dict[str, list] = {}
    for symbol in symbols:
        try:
            candles_by_symbol[symbol] = await provider.get_ohlcv(symbol, timeframe, limit=req.limit)
        except NotImplementedError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{provider.name} isn't wired up yet: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean 400, not a raw 500 traceback
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"Could not fetch {symbol} from {provider.name}: {describe_error(exc)}"
            ) from exc

    try:
        combos = [
            (symbol, strategy_name, STRATEGIES[strategy_name](params_by_strategy.get(strategy_name, {})))
            for symbol in symbols
            for strategy_name in req.strategies
        ]
    except ValueError as exc:  # e.g. the ml strategy without a trained model_path
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    weights_by_key = None
    if req.weights is not None:
        combo_keys = {combo_key(symbol, name) for symbol, name, _ in combos}
        if set(req.weights) != combo_keys:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"weights keys must be exactly {sorted(combo_keys)}"
            )
        if abs(sum(req.weights.values()) - 1) > 1e-6:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "weights must sum to 1")
        weights_by_key = {key: Decimal(str(value)) for key, value in req.weights.items()}
    runs, combined = run_portfolio(
        candles_by_symbol, combos, starting_balance=Decimal(str(req.balance)), weights=weights_by_key
    )

    return {
        "combined": _result_to_json(combined),
        "per_combo": [
            {
                "symbol": symbol,
                "strategy": strategy_name,
                "trades": result.total_trades,
                "win_rate": float(result.win_rate) if result.trades else None,
                "profit_factor": float(result.profit_factor) if result.profit_factor is not None else None,
                "pnl": float(result.total_pnl),
            }
            for symbol, strategy_name, result in runs
        ],
        "correlation": {
            symbol: {other: float(value) for other, value in row.items()}
            for symbol, row in correlation_matrix(candles_by_symbol).items()
        },
    }


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("tradon.web.app:app", host=settings.web_host, port=settings.web_port)


if __name__ == "__main__":
    main()
