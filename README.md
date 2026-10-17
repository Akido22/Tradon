# Tradon

Signal-only multi-market trading bot core (Forex first, crypto later). See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full concept and staged roadmap.

## Stage 1 (MVP, Forex, CLI) + first pass at Stage 3 (desktop GUI)

```
src/tradon/
├── config.py          # Settings (.env) + YAML app config loader
├── data/               # Data Layer: DataProvider interface + adapters
│   ├── base.py         #   Candle model, DataProvider ABC
│   ├── metatrader.py    #   Forex source (needs a running MT5 terminal)
│   ├── yahoo.py          #   reserve adapter (not wired up)
│   └── binance.py, bybit.py      # Stage 2 (crypto) adapters
├── strategy/            # Strategy Engine: plugins + indicator library
│   ├── base.py, indicators.py
│   └── plugins/sma_cross.py, rsi_strategy.py
├── signal_engine/        # Aggregation, dedup, confirmation, event bus
├── risk/                 # Position sizing / SL / TP (independent of strategy)
├── db/                    # SQLAlchemy models + Alembic migrations
├── scheduler/             # APScheduler wrapper (used by LiveSession's polling fallback)
├── runtime.py             # Shared core: LiveSession (realtime/poll), wires Data/Strategy/Signal + DB
├── cli/                   # Typer + Rich — calls runtime.py, renders as a table
└── gui/                   # PySide6 + qasync — calls runtime.py, renders as a window
```

`runtime.py` exists so CLI and GUI never duplicate orchestration logic — per
ARCHITECTURE.md §3.7, a presentation layer only calls the core and renders
the result. Adding Web (Stage 4) or bots (Stage 5) later means writing
another thin caller against `runtime.py`, not touching this module.

## Setup

1. **Python 3.11+ virtualenv** (already created in this repo as `.venv`):

   ```bash
   python -m venv .venv
   .venv/Scripts/pip install -e ".[dev]"
   ```

   > If pip fails with `Missing dependencies for SOCKS support`, a system-wide
   > SOCKS proxy is registered in Windows (`HKCU\...\Internet Settings`) and
   > `PySocks` isn't installed yet. Work around it for any `pip install`
   > (applies to `.[gui]` too):
   > - PowerShell: `$env:NO_PROXY='*'; .venv\Scripts\pip install -e ".[dev]"`
   > - bash: `NO_PROXY='*' .venv/Scripts/pip install -e ".[dev]"`

2. **Database — SQLite by default, nothing to install or run.** Just apply
   the migration and a `data/tradon.db` file is created automatically:

   ```bash
   .venv/Scripts/alembic upgrade head
   ```

   Postgres (Docker locally, or a VPS over SSH) is still available if you
   outgrow single-machine use — see "Database setup" below.

3. **Configure secrets**: copy `.env.example` to `.env`. `config.yaml`'s
   default provider is `bybit` (public crypto market data, no API key
   needed) so the CLI works right away without this step. For Forex, use
   `metatrader` instead — no API key, but needs the `metatrader` extra
   (`pip install -e ".[metatrader]"`, Windows only) and a running,
   logged-in MT5 terminal on the same machine (it talks to the terminal
   over local IPC, not the network).
   Non-secret settings (instruments, strategies, poll interval) live in
   `config/config.yaml`.

4. **Run the CLI**:

   ```bash
   .venv/Scripts/tradon run                          # continuous monitoring (config.yaml instruments), Ctrl+C to stop
   .venv/Scripts/tradon run --symbols BTCUSDT,ETHUSDT # override instruments for this run
   .venv/Scripts/tradon run --symbols EURUSD,GBPUSD --provider metatrader  # Forex, needs a running MT5 terminal
   .venv/Scripts/tradon run --once                    # single poll cycle then exit
   .venv/Scripts/tradon symbols                        # list symbols from the configured provider
   ```

5. **Run the GUI** (optional — needs the `gui` extra):

   ```bash
   .venv/Scripts/pip install -e ".[gui]"
   .venv/Scripts/tradon-gui
   ```

   Sidebar checkboxes pick which instruments/strategies are active (default
   from `config.yaml`); "Add pair" adds a symbol not in the config; the "..."
   button next to a strategy edits its parameters (periods/thresholds).
   Toolbar picks the data source (same registry as `tradon run --provider`)
   and which checked instrument the candlestick chart tracks. Start
   validates the checked instruments against the provider before polling
   begins (scheduled via APScheduler, same as the CLI — see `runtime.py`).
   Closing the window or clicking Stop cancels the schedule cleanly.

   On launch it loads recent signal history from the DB (not just what
   happens live in this session). A "Backtest" button opens a dialog to
   backtest any instrument/strategy combo without leaving the GUI — same
   `run_backtest()` the CLI uses. Instrument/strategy selection, params, poll
   interval, and provider are remembered between launches (`QSettings`, not
   `config.yaml` — GUI preferences don't overwrite the shared bot config).

## Web dashboard (optional)

FastAPI + server-rendered Jinja2 templates (no separate frontend build —
"проще для старта" per ARCHITECTURE.md §4). Same rule as CLI/GUI/Telegram:
runs its own `PollSession` against `runtime.py`; the browser gets history
via REST and live updates via WebSocket, per §2 Этап 4.

```bash
.venv/Scripts/pip install -e ".[web]"
.venv/Scripts/tradon create-user --email you@example.com --password ...
.venv/Scripts/tradon-web
```

Open http://127.0.0.1:8000 — session-cookie login against the `users`
table (no public signup; `create-user` is how accounts get made, so
multiple people can each have their own login — see ARCHITECTURE.md §2
Этап 4 "многопользовательский доступ"). Dashboard shows a candlestick chart
per instrument and a signal feed that loads history from `/api/signals`
then updates live over `/ws/signals`. `WEB_SECRET_KEY` in `.env` signs the
session cookie — the default is fine for localhost-only use.

**Portfolio analytics** (`/portfolio`, linked from the dashboard header):
pick any combination of instruments and strategies, backtest them all via
`tradon.backtest.run_portfolio()`, and see combined win rate / profit factor
/ max drawdown, an equity curve chart, and a per-combo breakdown table — the
same math as `tradon backtest --symbol A,B --strategy X,Y`, just interactive
instead of a one-shot CLI table. The per-combo rows are each backtested in
isolation (full starting balance each, "how would this one do alone"); the
combined number is a real shared-capital simulation (`simulate_portfolio()`)
— every combo trades against ONE account on a single merged timeline, sized
off its capital share (`weights`, equal split by default) of the *current*
balance, so a drawdown in one instrument genuinely shrinks what another can
risk next, same as a real multi-instrument account. `POST
/api/portfolio-backtest` accepts an optional `weights` object keyed
`"SYMBOL/strategy"` (must sum to 1) to override the equal split.

Metrics include Sharpe and Sortino ratios (annualized from per-trade
returns — an approximation, since trades close at irregular intervals
rather than on a fixed calendar schedule) alongside win rate / profit
factor / drawdown. Below the per-combo table is a correlation matrix of
the picked instruments' close-to-close returns — two instruments
correlated near +1 carry largely the same risk twice over, so running the
same directional strategy on both isn't real diversification even though
it looks like two positions.

## Realtime vs polling

`LiveSession` (used by CLI/GUI/Telegram/Web — they never poll on their own)
warms up history with one REST call per instrument, then tries the
provider's realtime WebSocket stream (`subscribe_realtime`). If the
provider doesn't support it (Yahoo) or the setup doesn't fit its realtime
limits (a free-tier subscription cap with more instruments configured
than it allows, for instance), it falls back to REST polling on
`poll_interval_seconds` —
automatically, no config needed either way. A dropped WebSocket reconnects
with backoff instead of falling back, since realtime is known to work for
that setup by then.

Realtime candles only arrive when a bar actually closes, so the first live
signal can take up to one full timeframe interval to show up even on a
healthy connection — unlike polling, which re-evaluates strategies against
the still-forming latest candle every tick. CLI prints the mode once it
settles; GUI shows it in the connection label; Web exposes it at
`/api/status` (polled by the dashboard's mode pill); Telegram's `/status`
includes it.

## Risk suggestions on live signals

Every live signal (CLI, GUI, Telegram) is run through the same `RiskManager`
the backtester uses — Size/Stop/Target columns (or a "Suggested:" line in
Telegram) show a position size and stop-loss/take-profit against a
hypothetical account balance. This is advisory only, not a real order (§0/§6)
— there's no broker account behind it. Tune it under `risk:` in
`config.yaml` (`account_balance`, `risk_per_trade`, `atr_multiplier`,
`reward_risk_ratio` — same defaults as `RiskManager()`). Persisted on the
`signals` table (`size`/`entry`/`stop_loss`/`take_profit`, nullable — a
signal can fire before there's enough history for ATR).

## Database setup

**Default: SQLite.** `tradon.config.Settings.database_url` points at
`data/tradon.db` (gitignored) unless `DATABASE_URL` is set in `.env`. No
server, no Docker, no separate process — CLI and GUI just open the same
file, so either one sees history the other wrote. WAL mode + a 5s busy
timeout are set on every connection (`db/session.py`) so CLI and GUI running
at the same time don't hit "database is locked"; still, for now, expect to
run one interface at a time rather than both polling simultaneously.

Same models, same migration, same code either way: switching to Postgres
later is a `DATABASE_URL` change only, nothing else.

**Alternative: Postgres**, for when single-machine SQLite stops being
enough (Stage 4 Web, multiple users):

- Locally via Docker: `docker compose --profile postgres up -d db`, then
  point `DATABASE_URL` at `postgresql+asyncpg://bot:bot@localhost:5432/trading_bot`.
- On a VPS over an SSH tunnel (keeps port 5432 off the public internet):

  ```
  ssh youruser@your-vps-ip
  sudo apt update && sudo apt install -y postgresql
  sudo -u postgres psql -c "CREATE USER bot WITH PASSWORD 'pick-a-password';"
  sudo -u postgres psql -c "CREATE DATABASE trading_bot OWNER bot;"
  ```

  Then, on Windows, open the tunnel (leave it running while using the bot)
  and point `DATABASE_URL` at `localhost` same as the Docker case:

  ```bash
  ssh -N -L 5432:localhost:5432 youruser@your-vps-ip
  ```

Either way, apply migrations once `DATABASE_URL` is set:

```bash
.venv/Scripts/alembic upgrade head
```

## Docker deployment (optional, for the Web dashboard / Telegram bot on a VPS)

Packages the headless server-side pieces only — GUI (PySide6) and
MetaTrader5 are Windows-desktop-only and never belong in a container, so
the image installs the `web`/`telegram` extras (`ml` is opt-in, see
below). SQLite is the default, same as running locally; nothing about the
database changes just because it's containerized.

```bash
cp .env.example .env   # fill in real values — WEB_SECRET_KEY especially, for anything beyond a trusted network
docker compose up -d --build   # starts `web` on :8000; migrations run automatically on every boot
```

- `./data` and `./config` are bind-mounted into the container, so the
  SQLite file and `config.yaml` persist and are editable from the host
  without rebuilding the image.
- The Telegram bot is a separate optional service (it runs its own
  independent live session, same as it would outside Docker):
  `docker compose --profile telegram up -d telegram`.
- Postgres instead of SQLite: `docker compose --profile postgres up -d db`,
  then set `DATABASE_URL` in `.env` (see "Database setup" above) and
  restart `web`.
- To bundle the ML strategy's dependencies (scikit-learn/scipy) into the
  image: `docker build --build-arg EXTRAS=web,telegram,ml -t tradon .`,
  then `docker compose up -d` as usual (compose reuses the image you just
  built under the same tag/service name).

**A gotcha this caught when first built**: `tradon-web` used to hardcode
`host="127.0.0.1"` for uvicorn — fine outside a container, but inside one
that binds only the container's own loopback, which a published port
(`-p 8000:8000`) cannot forward into; the dashboard would be completely
unreachable from outside despite the port mapping looking correct. Fixed
via a `WEB_HOST` setting (defaults to `127.0.0.1` for local dev, compose
sets it to `0.0.0.0`) — mentioned here since it's the kind of thing that
looks like it should work and silently doesn't.

## Telegram notifications (optional)

Another presentation layer, same pattern as CLI/GUI — runs its own
`PollSession` against `runtime.py` and pushes each published signal to one
chat. No analysis logic here either.

```bash
.venv/Scripts/pip install -e ".[telegram]"
.venv/Scripts/tradon-telegram
```

1. Create a bot with [@BotFather](https://t.me/BotFather), put the token in
   `.env` as `TELEGRAM_BOT_TOKEN`.
2. Run `tradon-telegram`, send the bot `/start` — it replies with your chat
   id. Put that in `.env` as `TELEGRAM_CHAT_ID`, restart.
3. `/status` shows what it's currently watching (instruments, provider, poll
   interval — from `config.yaml`, same as CLI/GUI).

## Backtesting

Replays a strategy over historical candles using the same `Strategy`/
`RiskManager` code the live poll loop uses — no separate "backtest strategy",
and no orders placed (signal-only, see ARCHITECTURE.md §0/§6):

```bash
.venv/Scripts/tradon backtest --symbol BTCUSDT --strategy sma_cross --provider binance --timeframe 1h --limit 1000
.venv/Scripts/tradon backtest --symbol EURUSD --strategy rsi --provider metatrader --limit 500
```

Reports win rate, profit factor, max drawdown, total PnL. Strategy params
come from `config.yaml` if that strategy is listed there, else the plugin's
own defaults.

**Portfolio mode**: pass comma-separated `--symbol`/`--strategy` to backtest
every combination. The per-combo table still isolates each one (full
`--balance` each, so you can see standalone quality); "Portfolio (shared
capital)" is a real simulation — every combo trades on one shared account
along a single merged timeline, sized off its share of the *current* balance
(equal split by default, override with `--weights`):

```bash
.venv/Scripts/tradon backtest --symbol BTCUSDT,ETHUSDT --strategy sma_cross,rsi --provider binance
.venv/Scripts/tradon backtest --symbol BTCUSDT,ETHUSDT --strategy sma_cross,rsi --provider binance --weights 0.4,0.4,0.1,0.1
```

**Parameter search**: grid-search a strategy's parameters and rank by total
PnL — useful before trusting any strategy's defaults (ours weren't
profitable out of the box on BTCUSDT/1h — see `_OPTIMIZE_GRIDS` in
`cli/main.py` to change the search ranges):

```bash
.venv/Scripts/tradon optimize-params --symbol BTCUSDT --strategy sma_cross --provider binance --timeframe 1h --limit 1000
```

**Walk-forward validation**: `optimize-params` grades every combo on the
same window it's picked from, so it will always surface *something* that
happened to fit that window's noise — the PnL it reports is optimistic by
construction. `walk-forward` splits history into consecutive folds, re-runs
the grid search on each fold using only data before it, and backtests
*only* the chosen params on the fold after (data the search never saw),
chaining every fold's out-of-sample result into one real equity curve. Needs
more history than `optimize-params` since it gets split into folds:

```bash
.venv/Scripts/tradon walk-forward --symbol BTCUSDT --strategy sma_cross --provider binance --timeframe 1h --limit 2000 --splits 5
```

## ML strategy (optional)

Optional extra — the `ml` strategy needs `pip install -e ".[ml]"`
(scikit-learn + joblib) and, unlike every other strategy, doesn't run with
defaults: it needs a model trained and saved first. Training is a separate
offline step (`tradon train-model`), not something that happens inside
`on_candles()` — the strategy at runtime just loads the saved model and
calls `predict_proba()`, the same "train offline, infer online" split any
real ML deployment uses.

```bash
.venv/Scripts/tradon train-model --symbol BTCUSDT --provider binance --timeframe 1h --limit 3000 --horizon 5 --model-path models/btcusdt_1h.joblib
```

This fits a gradient-boosted classifier on engineered features (RSI, MACD
histogram, SMA ratio, ATR%, Bollinger-band position, short-term returns —
see `tradon/ml/features.py`) to predict whether price will be higher
`--horizon` bars ahead, using a **chronological** train/test split (the
most recent `--test-fraction` held out, never shuffled in — shuffling
would let the model train on data from after the point it's graded on,
which is look-ahead bias in disguise). It prints train/test accuracy next
to the test period's majority-class baseline — if test accuracy doesn't
beat that baseline, the model has no real edge on this data and printing
a warning says so; don't enable it for trading just because a number came
out.

**Multiple symbols**: pass a comma-separated `--symbol` list
(`--symbol BTCUSDT,ETHUSDT`) to pool their history into one dataset, so
the model learns patterns that hold across instruments instead of one
symbol's idiosyncrasies. Each symbol is split chronologically on its
own — its own earliest bars for train, its own latest for test — *before*
pooling, since two different symbols' bars at "the same relative
position" aren't the same point in time; splitting after pooling
wouldn't actually guarantee every training row precedes every test row
the way a real look-ahead-safe split requires. Confirmed live: pooling
BTCUSDT+ETHUSDT (1000 candles each) produced 1870 samples, double a
single symbol's ~935 — rows from both genuinely went in, not just one.
This isn't a silver bullet, though — in that same run neither the pooled
nor the single-symbol model beat its baseline; more data doesn't fix a
model that isn't finding a real edge.

To actually use a trained model, add it to `config.yaml`'s strategies list:

```yaml
  - name: ml
    enabled: false
    params:
      model_path: "models/btcusdt_1h.joblib"
      long_threshold: 0.55
      short_threshold: 0.45
```

`long_threshold`/`short_threshold` are the P(price higher) cutoffs that
turn a probability into a LONG/SHORT signal — between them, no signal.
`optimize-params`/`walk-forward` don't support `ml` (its real
hyperparameters require retraining, which doesn't fit the same
grid-search loop as an indicator's params) — tune thresholds by hand via
`tradon backtest --strategy ml`, or retrain with a different `--horizon`.

**Watch out for in-sample leakage when backtesting it manually.**
`get_ohlcv(limit=N)` always fetches the *most recent* N candles, so
training on `--limit 3000` and then immediately running `tradon backtest
--strategy ml --limit 2000` against the same symbol/timeframe mostly
re-tests the model on data it already trained on — the result will look
dramatically, misleadingly good (confirmed live: one such run showed
+80% PnL that dropped to roughly break-even once tested only on candles
strictly after the training cutoff). `train-model`'s own train/test split
is chronological and honest; a `backtest` run against overlapping recent
history is not. If you want to sanity-check a trained model against data
it's never seen, train on an older slice and hand-verify against a
strictly newer one (there's no CLI flag for this yet — `get_ohlcv` has no
date-range parameter, only `limit`).

## Tests

```bash
.venv/Scripts/pytest
```

Covers the core (`backtest.py`/`runtime.py`/`ml`) plus all three
presentation layers against a shared `FakeProvider` test double (see
conftest.py) — no network calls, no real exchange/broker involved:
- `test_cli.py` — real Typer command invocations (`symbols`, `run`,
  `backtest`, `optimize-params`, `walk-forward`, `train-model`), including
  error paths (unknown provider/strategy, unsupported symbol)
- `test_web_app.py` — a real FastAPI `TestClient` against the actual app
  (auth flow, session cookies, every `/api/*` endpoint, portfolio
  backtest)
- `test_gui.py` — optional (`pip install -e ".[gui]"`), builds `MainWindow`
  offscreen (`QT_QPA_PLATFORM=offscreen`) and exercises its synchronous
  methods directly (chart rendering, instrument/strategy checkboxes,
  provider switching). Deliberately does **not** drive the full async
  start/stop monitoring flow — doing that inside an automated offscreen
  test hit asyncio task re-entrancy errors specific to the test harness
  during manual exploration, and a flaky GUI test is worse than a
  narrower one that's actually reliable in CI.

## Notes

- **iTick was removed as a provider.** ARCHITECTURE.md originally called
  for it as the Forex-first source, and it was live-verified working
  early on, but its free-tier API key access stopped working in practice
  (`401 Unauthorized`, needing periodic manual renewal at itick.org) and
  wasn't worth keeping around for that. Forex now goes through
  `metatrader` instead; `provider`/`instruments` in `config.yaml` default
  to `bybit`/`BTCUSDT,ETHUSDT` so the CLI works without any provider setup
  at all. `tradon.data.itick` no longer exists — swap in a different Forex
  REST+WebSocket provider the same way if you need one (implement the
  `DataProvider` interface, register it in `PROVIDERS` in `runtime.py`).
