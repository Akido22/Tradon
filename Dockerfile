# Packages the headless server-side pieces (Web dashboard, Telegram bot,
# `tradon run`/backtest CLI) for deployment. GUI (PySide6) and MetaTrader5
# are Windows-desktop-only by nature and never belong in a container —
# this image only ever installs the `web`/`telegram`/`ml` extras.
FROM python:3.12-slim

WORKDIR /app

# No system build deps needed: every dependency this image installs
# (sqlalchemy, aiosqlite, fastapi, uvicorn, bcrypt, scikit-learn if the ml
# extra is requested) ships manylinux wheels for this base image, so a
# full build toolchain would only bloat the image for nothing.

COPY pyproject.toml alembic.ini ./
COPY src ./src
COPY config ./config

# Which optional extras to install — override at build time, e.g.
# `docker build --build-arg EXTRAS=web,telegram,ml .` to also bundle the
# ML strategy (pulls in scikit-learn/scipy, hence opt-in not default).
ARG EXTRAS=web,telegram
RUN pip install --no-cache-dir -e ".[${EXTRAS}]"

# SQLite's default data dir (see tradon/config.py) — created here so the
# app can write to it even before the ./data bind mount from compose lands
# on top of it; harmless either way.
RUN mkdir -p /app/data

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["tradon-web"]
