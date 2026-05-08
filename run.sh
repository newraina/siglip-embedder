#!/bin/bash
# Pure HTTP launcher. uvicorn loads the SigLIP model synchronously at import
# time (server.py), so /health stays unreachable until the model is on the
# GPU. Whatever orchestrator you run this under (k8s, plain docker run,
# Salad Container Group, RunPod, …) should size its startup probe / healthcheck
# accordingly.

set -e

PORT="${PORT:-8000}"

exec uvicorn server:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --log-level info
