#!/usr/bin/env bash
# InfraEdge-Shield one-command demo: backend + frontend + pipeline (forced attacks).
# Usage:
#   ./demo.sh                 # dummy feed, forced attack every 10th frame
#   ./demo.sh --no-demo       # normal detection (no forced attacks)
#   DEMO_EVERY=5 ./demo.sh    # attack every 5th frame
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO_EVERY="${DEMO_EVERY:-10}"
if [[ "$1" == "--no-demo" ]]; then DEMO_EVERY=0; fi

echo "=== InfraEdge-Shield Demo (attack every ${DEMO_EVERY:-off}) ==="
command -v mosquitto >/dev/null 2>&1 && (pgrep mosquitto >/dev/null || (mosquitto -d && echo "[demo] mosquitto started")) || echo "[demo] mosquitto not found — MQTT will be skipped (backend POST still works)"

cd "$ROOT/dashboard/backend" && (npm start >/tmp/opencode/demo-backend.log 2>&1 &) && echo "[demo] backend → http://localhost:3000 (log /tmp/opencode/demo-backend.log)"
cd "$ROOT/dashboard/frontend" && (npm run dev -- --port 5173 >/tmp/opencode/demo-frontend.log 2>&1 &) && echo "[demo] frontend → http://localhost:5173"
sleep 3
cd "$ROOT"
if [[ "$DEMO_EVERY" -gt 0 ]]; then
  venv/bin/python inference/pipeline.py --dummy --demo-attack-every "$DEMO_EVERY"
else
  venv/bin/python inference/pipeline.py --dummy
fi
