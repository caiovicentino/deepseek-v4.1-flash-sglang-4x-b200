#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on 4x B200 — the production command (2026-10-17).
#
# Recipe: SGLang DeepSeek-V4.1 cookbook + the B200 TP4/EP4 configuration validated in InferenceX PR #3346:
#   Engram in host RAM per rank, mem 0.80, chunk 4096 (do not raise), 16 decode rounds between prefill chunks,
#   SWA prefix tails 64 x CONC (<= 4096), at most 64 running requests, DSpark block 5, thinking on at effort high.
# Needs the two patches in ../patches (thinking rule; tool-result image order).
#
# Inside lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0:   API_KEY=... MODEL_PATH=... bash serve/serve.sh
# With the image extracted by tools/extract_image.py:           IMG=/root/dsv41img API_KEY=... bash serve/serve.sh
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/models/dsv41-1017}"
: "${API_KEY:?set API_KEY}"
CONC="${CONC:-64}"
TAILS=$(( 64 * CONC )); [ "$TAILS" -gt 4096 ] && TAILS=4096; [ "$TAILS" -lt 128 ] && TAILS=128

export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export NCCL_CUMEM_ENABLE=1 SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING="${THINKING:-1}" SGLANG_DSV41_REASONING_EFFORT="${EFFORT:-high}"

if [ -n "${IMG:-}" ]; then          # extracted image: its own venv and source tree
  PY="$IMG/opt/sglang/bin/python"; export PATH="$IMG/opt/sglang/bin:$PATH"; cd "$IMG/sgl-workspace/sglang"
else
  PY=python3
fi

# We serve under the previous model's name so existing client configs keep working; the server ignores the
# request's `model` field anyway. Set SERVED_MODEL_NAME to whatever your clients expect.
exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4.1-flash}" --trust-remote-code \
  --tp-size 4 --ep-size 4 \
  --mem-fraction-static "${MEMF:-0.80}" \
  --chunked-prefill-size "${CHUNK:-4096}" --prefill-decode-interval "${PDI:-16}" --swa-prefix-tails "$TAILS" \
  --max-running-requests "$CONC" --cuda-graph-max-bs-decode "$CONC" \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 --schedule-policy "${SCHED:-lpm}" \
  --enable-multimodal --image-processor-backend pil --enable-cache-report --enable-metrics \
  --media-url-max-file-size-mb 1024 \
  ${EXTRA_ARGS:-} --api-key "$API_KEY" --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
