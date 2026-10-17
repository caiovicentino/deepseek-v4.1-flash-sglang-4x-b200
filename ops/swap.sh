#!/usr/bin/env bash
# Swap a running SGLang server for DeepSeek-V4.1-Flash-1017, with rollback.
#
#   ROLLBACK=/path/to/your_previous_start.sh bash ops/swap.sh
#
# 1. stops whatever SGLang runs (by PID), waits until the GPUs are really free (the memory lags the process exit);
# 2. launches serve/serve.sh detached, logging to $LOG;
# 3. waits for /health = 200 (first boot ~17 min with FlashInfer autotuning, then ~7 min);
# 4. if it dies or does not become healthy in BOOT_S seconds and ROLLBACK is set, runs ROLLBACK.
# Pause your restart watchdog first (ours skips everything while /root/MANUTENCAO exists).
# Then run tests/val_dsv41.py and tests/val_leak.py before letting the watchdog back.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${LOG:-/root/dsv41.log}"; BOOT_S="${BOOT_S:-1800}"; URL="${URL:-http://127.0.0.1:8000/health}"
say() { echo "[$(date -u +%FT%TZ)] $*"; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '; }

for p in $(pgrep -f "sglang.launch_server"); do kill "$p" 2>/dev/null; done
sleep 10
for _ in $(seq 1 45); do [ -z "$(gpu_pids)" ] && break; sleep 2; done
for p in $(gpu_pids); do kill -9 "$p" 2>/dev/null; done
for _ in $(seq 1 30); do [ -z "$(gpu_pids)" ] && break; sleep 2; done
say "GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"

[ -s "$LOG" ] && mv "$LOG" "$LOG.pre-$(date -u +%Y%m%d-%H%M%S)"
setsid nohup bash "$HERE/serve/serve.sh" >> "$LOG" 2>&1 < /dev/null &
say "launched; waiting for /health"
t=0
while [ "$t" -lt "$BOOT_S" ]; do
  sleep 15; t=$((t + 15))
  if [ "$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$URL")" = "200" ]; then say "healthy after ${t}s"; exit 0; fi
  if ! pgrep -f "sglang.launch_server" >/dev/null; then say "died during boot"; tail -30 "$LOG"; break; fi
done
say "not healthy"
if [ -n "${ROLLBACK:-}" ]; then say "rolling back with $ROLLBACK"; bash "$ROLLBACK"; fi
exit 1
