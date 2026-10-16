#!/bin/sh
# Applies pending Alembic migrations before every start — idempotent (a
# no-op once the schema is current), so it's safe to run on every
# container boot rather than requiring a separate manual migration step
# for Docker deployments. Then execs whatever command compose/CMD passed
# (tradon-web, tradon-telegram, or an ad-hoc `tradon ...` one-off).
set -e
alembic upgrade head
exec "$@"
