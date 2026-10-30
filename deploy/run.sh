#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AUREL_EC_CONFIG_ROOT:?Set the private configuration directory}"
: "${AUREL_EC_STATE:?Set an isolated private state directory}"
exec "${AUREL_EC_PYTHON:-python3}" -m uvicorn aurel_ec.service:app_factory \
  --factory --host "${AUREL_EC_HOST:-127.0.0.1}" --port "${AUREL_EC_PORT:-18086}" \
  --workers 1 --no-access-log
