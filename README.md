# DeepSeek-V4.1-Flash-1017 on 4× NVIDIA B200 with SGLang

Production recipe for serving **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
(MIT; 40 layers, 384 routed experts with 6 active, FP4 experts + FP8 attention, Engram n-gram tables, vision,
1M context, DSpark speculative decoding) on **four B200** with SGLang, for the same community workload that ran
GLM-5.3-Flash on this machine: agentic coding traffic with very long prompts and a high prefix-cache hit rate.

> **Status — 2026-10-18**
> - **In production since 2026-10-17 11:08 UTC**, replacing GLM-5.3-Flash-NVFP4 (see the
>   [GLM 4× B200 repo](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200)) on the same machine.
>   The checkpoint was published ~8 h earlier; we validated it the same day (below).
> - **Functional validation passed** (16/16 checks + 5/5 reasoning-leak checks): kernel-boundary prompts from 60 to
>   14k tokens, a 3,200-token Portuguese generation with no anomalous characters, greedy code, tool call and tool
>   result round trip, image input, reasoning effort, the request shapes our clients send, streaming.
> - **One server-side bug found and patched** before it reached users for long: on the nightly image, a request that
>   says nothing about thinking got its raw reasoning and a `</think>` inside `message.content`
>   ([`patches/`](patches/)).
> - **24-hour production measurement** (2026-10-17 11:21 → 2026-10-18 11:21 UTC, against GLM-5.3-Flash v3.6 on the
>   same hours of the two previous days; [`docs/results.md`](docs/results.md) §3): decode per user 1.9× GLM's at 1–8
>   running requests and 1.5× at 9–16 (510 / 346 / 169 tok/s at 1 / 2–8 / 9–16); time to first token p50 / p90 / p99
>   0.39 / 1.45 / 6.0 s (GLM 0.99 / 4.0 / 14.7 s; the DeepSeek day had 10% fewer requests and less uncached prompt, so
>   part of this gap is load); prefix-cache hit 96.7%. The busiest bands (17–64 running) could not
>   be compared: traffic never put more than 20 requests in flight. Under traffic, prompts are admitted ~20% slower
>   than on GLM (16 decode steps between prefill chunks); that interval is the first tuning change, one at a time.

Series: [GLM-5.3-Flash 4× RTX PRO 6000](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-rtx-pro-6000) ·
[GLM-5.3-Flash 2× B200](https://github.com/caiovicentino/glm-5.3-flash-sglang-2x-b200) ·
[GLM-5.3-Flash 4× B200](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200) · this repo.

## Quick start

On a host with Docker:

```bash
git clone https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200 && cd deepseek-v4.1-flash-sglang-4x-b200
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /models/dsv41-1017          # 510 GB
# image: lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0 (no released SGLang serves V4.1 yet)
# inside the container: apply the two patches, then serve
python patches/patch_thinking_dsv41.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
bash patches/apply_encoding_fix.sh /sgl-workspace/sglang
export API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')
MODEL_PATH=/models/dsv41-1017 bash serve/serve.sh
python tests/val_dsv41.py && python tests/val_leak.py     # once /health answers 200
```

Without Docker (our case: a Vast.ai container already running an older SGLang image), extract the nightly image into
a separate directory with [`tools/extract_image.py`](tools/extract_image.py) and point `serve/serve.sh` at it with
`IMG=/root/dsv41img`. The production SGLang install is never touched, so going back is one script
([`ops/`](ops/)).

The first boot took **17 minutes** (weights, Engram host tables, then FlashInfer autotuning of the MXFP8 GEMMs);
later boots took **3.3–4.4 minutes** from launch to ready with the autotune cache (5–7 minutes with stopping the old
server and the checks). Warm the server before measuring.

## Recipe card (production, 2026-10-17)

| | |
|---|---|
| hardware | 4× B200 (179 GiB each, NVLink NV18 all-to-all), Xeon Platinum 8559C, 2 TB RAM, NVMe; driver 595.91 (CUDA 13.2), image built for CUDA 13.0 — a Vast.ai datacenter offer at ~US$ 28/h |
| image | `lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0`: SGLang a93f16f0, torch 2.14.1+cu130, sgl-kernel 0.4.10, FlashInfer 0.7.1, DeepGEMM; Python 3.12.15. Plus the two patches in [`patches/`](patches/) |
| checkpoint | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): 510 GB; FP8 (e4m3, 32×32 blocks, ue8m0 scales) attention and dense, **FP4 routed experts**, two Engram tables (layers 1 and 14, 384M rows each), a 32-layer vision encoder |
| parallelism | `--tp-size 4 --ep-size 4` |
| Engram | `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`: the n-gram tables stay in host RAM, sharded per rank, so HBM goes to weights and KV. Weights take 74.5 GiB per GPU, plus 2.1 GiB for the DSpark head |
| attention / MoE | chosen by SGLang for this checkpoint: DSv4 attention backend (page size 256), KV cache fp8_e4m3, `flashinfer_mxfp4` MoE runner, `flashinfer_cutedsl` for the MXFP8 dense GEMMs |
| speculation | `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` (the checkpoint's own DSpark head; 5 is the structural maximum) |
| memory | `--mem-fraction-static 0.80` → **KV pool ~20.0 M tokens** (20,019,200 or 19,998,976 depending on the boot), 41 GiB still free per GPU after allocation at boot; under traffic the server's allocator takes most of it (5–6.4 GiB free). Context 1,048,576 (the model's default) |
| prefill | `--chunked-prefill-size 4096 --prefill-decode-interval 16`. **Do not raise the chunk**: the sparse-attention indexer and DSpark prefill buffers scale with chunk × context (we hit an indexer OOM at 8192 with V4.1 on another GPU) |
| prefix cache | `--swa-prefix-tails 4096` (= 64 × max running requests, capped at 4096): chunked requests leave reusable sliding-window tails in the radix tree |
| concurrency | `--max-running-requests 64 --cuda-graph-max-bs-decode 64`. In the [InferenceX sweep](https://github.com/SemiAnalysisAI/InferenceX/pull/3346) on the same topology, throughput collapsed at 128 concurrent requests |
| scheduling | `--schedule-policy lpm` (longest cached prefix first) |
| parsers | `--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41` |
| thinking | `SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV41_REASONING_EFFORT=high`: thinking on by default at effort *high*, as the cookbook recommends. Requires the thinking patch |
| images | `--enable-multimodal --image-processor-backend pil` (pre-processing on the CPU; on GLM the GPU path ran out of memory on GPU 0 at peak) |
| other | `NCCL_CUMEM_ENABLE=1`, `SGLANG_TIMEOUT_KEEP_ALIVE=900`, `--enable-cache-report --enable-metrics` |

The full command is [`serve/serve.sh`](serve/serve.sh). Sources: the SGLang
[DeepSeek-V4.1 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1) and the B200 TP4/EP4
configuration validated by SemiAnalysis in
[InferenceX PR #3346](https://github.com/SemiAnalysisAI/InferenceX/pull/3346) (GSM8K 97.0–97.9%).

## The two patches

1. **One thinking rule for the encoder and the reasoning parser** ([`patches/patch_thinking_dsv41.py`](patches/patch_thinking_dsv41.py)).
   On the 2026-10-16 nightly, the V4.1 encoder decides thinking from `chat_template_kwargs.thinking` or, when absent,
   `SGLANG_DEFAULT_THINKING`; the reasoning parser decides from a different rule that only separates reasoning when the
   request asks explicitly. With thinking on by default, every request that said nothing about thinking got the raw
   reasoning and `</think>` in `message.content`. The patch gives both sides one rule: `thinking` if present, else
   `enable_thinking` (accepted as an alias, because GLM-era clients send it), else the server default.
   **Our 16-check validation passed while the bug was live** — it was found by reading the outputs; that is why
   [`tests/val_leak.py`](tests/val_leak.py) exists.
2. **Tool-result image order and current-turn thinking** ([`patches/apply_encoding_fix.sh`](patches/apply_encoding_fix.sh)):
   `encoding_dsv41.py` from upstream SGLang commit
   [a038a0dca](https://github.com/sgl-project/sglang/commit/a038a0dca) (#44751), merged after the nightly. The script
   checks that the nightly's file is the commit's parent before replacing it.

## For client authors

- Thinking: `chat_template_kwargs: {"thinking": false}` (or `enable_thinking: false`) turns it off; the default is on.
- `reasoning_effort`: `low` / `high` / `xhigh` / `max`, or an integer budget 1–100 via `chat_template_kwargs`.
- Tool-call history: send the assistant message back with its `reasoning_content` next to `tool_calls`; without it the
  model tends to stop before acting.
- The server ignores the `model` field of a request: any model name is served by the same model.

## Repository

| path | what |
|---|---|
| [`serve/serve.sh`](serve/serve.sh) | the production command, all flags and environment |
| [`patches/`](patches/) | the two patches above |
| [`tools/extract_image.py`](tools/extract_image.py) | extracts an SGLang image into a directory without Docker (OCI whiteouts handled) |
| [`ops/`](ops/) | swap from a running server with rollback, restart, and the notes for a restart watchdog |
| [`tests/`](tests/) | `val_dsv41.py` (16 functional checks) and `val_leak.py` (reasoning never in `content`) |
| [`docs/results.md`](docs/results.md) | validation results and the 24-hour production measurement; its numbers in [`docs/data/results_24h.json`](docs/data/results_24h.json) |
| [`tools/measure24h/`](tools/measure24h/) | the read-only scripts that produced the 24-hour measurement |

MIT licensed, like the model.
