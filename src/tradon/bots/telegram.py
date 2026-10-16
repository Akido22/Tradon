"""Telegram bot — Presentation Layer (see ARCHITECTURE.md §3.7, Этап 5).

Same rule as CLI/GUI: no analysis logic. Runs its own LiveSession against
tradon.runtime (identical to what CLI's `run` and the GUI's Start button do)
and pushes every published signal to one configured chat. `aiogram` per the
tech stack in ARCHITECTURE.md §4.
"""
from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from tradon.config import get_settings
from tradon.runtime import (
    LiveSession,
    SignalWithPlan,
    build_provider,
    build_risk_manager,
    build_signal_engine,
    build_strategies,
    describe_error,
    load_runtime_config,
)

dp = Dispatcher()
live_session: LiveSession | None = None


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        f"Chat id: {message.chat.id}\n\n"
        "Add this to .env as TELEGRAM_CHAT_ID and restart the bot to start "
        "receiving signals in this chat."
    )


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    cfg = load_runtime_config()
    mode = live_session.mode if live_session is not None else "idle"
    await message.answer(
        f"Watching: {', '.join(cfg.instruments) or '(none configured)'}\n"
        f"Provider: {cfg.provider_name}\n"
        f"Mode: {mode}\n"
        f"Poll interval: {cfg.poll_interval_seconds}s (used only if mode is 'poll')"
    )


def format_signal(signal_with_plan: SignalWithPlan) -> str:
    signal, plan = signal_with_plan.signal, signal_with_plan.plan
    text = (
        f"{signal.symbol} — {signal.strategy}\n"
        f"{signal.direction.value.upper()} (strength {signal.strength:.2f})\n"
        f"{signal.reason}"
    )
    if plan is not None:
        text += f"\n\nSuggested: size {plan.size:.4f}, stop {plan.stop_loss:.5f}, target {plan.take_profit:.5f}"
    return text


async def _run() -> None:
    global live_session
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")

    bot = Bot(token=settings.telegram_bot_token)
    cfg = load_runtime_config()
    provider = build_provider(cfg)
    strategies = build_strategies(cfg)
    signal_engine = build_signal_engine(cfg)
    risk_manager = build_risk_manager(cfg)

    async def on_signals(signals: list[SignalWithPlan]) -> None:
        if not settings.telegram_chat_id:
            return
        for signal in signals:
            await bot.send_message(settings.telegram_chat_id, format_signal(signal))

    async def on_error(exc: Exception) -> None:
        if settings.telegram_chat_id:
            await bot.send_message(settings.telegram_chat_id, f"Polling error: {describe_error(exc)}")

    live_session = LiveSession(
        provider, cfg.instruments, cfg.timeframe, strategies, signal_engine,
        risk_manager, cfg.risk_account_balance,
        on_signals=on_signals, on_error=on_error,
    )
    if cfg.instruments:
        live_session.start(cfg.poll_interval_seconds)

    try:
        await dp.start_polling(bot)
    finally:
        live_session.stop()
        await live_session.wait_closed()
        await bot.session.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
