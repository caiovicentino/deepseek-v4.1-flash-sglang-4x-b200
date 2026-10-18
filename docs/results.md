# Results

## 1. Day-0 functional validation (2026-10-17, production server, real traffic running)

`tests/val_dsv41.py` and `tests/val_leak.py`, after the thinking patch. Before the patch the first script also
passed 16/16 while every request without a thinking setting leaked its reasoning into `message.content`; the second
script was written to catch exactly that.

| check | result |
|---|---|
| kernel boundaries: a code word at the start of prompts of 60, 84, 156, 276, 1,428, 5,220 and 14,028 tokens | 7/7 recalled |
| 3,200-token essay in Portuguese, anomalous characters per 2,000-character block | 0 in every block; coherent ending |
| greedy code (`temperature=0`), executed against a test | pass |
| tool call (name and JSON arguments) and the tool-result round trip with `reasoning_content` kept | pass, pass |
| image input (a solid red PNG) | "vermelho" |
| `reasoning_effort` low and high | correct answer, reasoning separated |
| GLM-era payload `chat_template_kwargs.enable_thinking=false` | clean answer, no reasoning |
| streaming | pass |
| reasoning never in `content` — default, default streaming, `enable_thinking=false`, `thinking=false`, effort low | 5/5 |

Boot facts (SGLang's boot log; its "GB" are GiB): weights 74.5 GiB per GPU plus 2.1 GiB for the DSpark head; KV and
SWA pools 52 GiB; CUDA graphs 5.5 GiB; 41 GiB free per GPU after allocation, most of which the running server then
takes (5–6.4 GiB free in nvidia-smi under traffic, flat; §3). KV pool ~20.0 M tokens (20,019,200 or 19,998,976
depending on the boot). Engram tables in host RAM. First boot 17 min; later boots 3.3–4.4 min from launch to ready
(5–7 min with stopping the old server and the checks).

## 2. 24-hour production measurement — protocol (written before the data, 2026-10-17)

Window: 24 hours from 2026-10-17 11:21 UTC (the patched server's start), on real community traffic.
Comparison: GLM-5.3-Flash v3.6 on the same machine over the same hours of the previous days available in our logs.
Nothing is changed on the server during the window, except a restart the watchdog may need (reported if it happens).
(Two restarts happened that the watchdog did not make; see "What was not as planned" in §3.)

What will be reported, and how:

1. **Decode speed per user by batch-size band** (1, 2–8, 9–16, 17–32, 33–64), from the server's per-step decode log
   lines, computed with the same script on both models' logs.
2. **Time to first token** p50/p90/p99, from deltas of the server's TTFT histogram sampled every 5 minutes (raw
   per-request timing is not logged).
3. **Prefill throughput** p50/p90 over full 4,096-token chunks only (partial chunks understate it).
4. **Speculation acceptance** (DSpark accept length and rate) by band.
5. **Prefix-cache hit rate**, requests and tokens per hour, peak concurrency, queueing.
6. **Errors**: HTTP non-200 at the gateway, server exceptions, restarts.
7. **Memory**: free GPU memory over the day, KV and SWA pool use.

Then tuning, one change per window, each against the same measurement: the prefill/decode interval (16 vs lower,
for time to first token), and DP attention with MegaMoE for the high-concurrency hours.

## 3. 24-hour results

Measured 2026-10-18 by the protocol in section 2, read-only on the server, with the scripts in
[`tools/measure24h/`](../tools/measure24h/) (one pass over the logs and the 5-minute metrics series; the same code
reads both models' logs). Every number below is in [`data/results_24h.json`](data/results_24h.json).

**Windows.** DeepSeek-V4.1-Flash-1017: 2026-10-17 11:21 → 2026-10-18 11:21 UTC (Saturday → Sunday). GLM-5.3-Flash
v3.6, same machine: the most recent full 24 hours it served, 2026-10-16 10:49:35 → 2026-10-17 10:49:35 UTC
(Friday → Saturday, ending when the swap stopped it), and the 24 hours before that (Thursday → Friday) to show how
much a GLM number moves from one day to the next. The hours of day are the same, shifted by 32 minutes.

**What was not as planned.** Section 2 said nothing would change during the window except a watchdog restart. There
were two restarts, neither by the watchdog (which restarted nothing):

| UTC, 2026-10-17 | | excluded |
|---|---|---|
| 17:23:48 → 17:40:01 | the container was restarted from outside (an orderly SIGTERM; cause not identified). The host's boot script still started an old GLM build, which served traffic 17:30–17:36; DeepSeek was back at 17:40:01. The boot script now starts DeepSeek | everything in the gap, including the GLM-served requests |
| 18:03:03 → 18:08:14 | planned restart: the HTTP server now binds to 127.0.0.1 (clients come in through the tunnel) and its output goes through a filter that masks request content in error messages. Every other server argument is identical before and after (checked by diffing the logged `server_args`) | the gap |

The metrics series covers 23.5 of the 24 hours (the gaps, plus the 5-minute intervals that span a restart).

**The two servers are configured differently**, so this compares production recipes, not models alone: both TP4/EP4,
4,096-token chunks, longest-prefix-first scheduling and one decode log line per 40 steps; DeepSeek runs
`--prefill-decode-interval 16`, at most 64 requests, DSpark with 5-token blocks and a 1M context; GLM v3.6 ran
interval 2, at most 96 requests, EAGLE with 3 draft steps (2 at 40+ requests) and a 524k context.

**Internal load.** Three client addresses appear only while our own studies were running (DeepSeek as an agent):
2,126 requests, 2.0% of the window, mostly 12:00–16:00 and 21:00–22:00 UTC on 2026-10-17, with short prompts
(1k–5k prompt tokens per run in the study logs). They are not removed: the per-step log lines carry no client.
Our study runs of 2026-10-18 started at 11:31 UTC, after the window. In the GLM windows none of these addresses
appear (one request), nor our old data-generation bench; unknown internal clients cannot be ruled out.

**Load.** The DeepSeek window had 10% fewer requests than the GLM day (4,564 vs 5,063 per hour) and fewer prompt
tokens per request (73k vs 96k, of which 2.4k vs 3.8k not cached; the two tokenizers differ, so token counts are
only roughly comparable). It also had far fewer requests in flight: 2 at the median 5-minute snapshot against 11.
Most of that drop follows from the faster service rather than from lighter traffic: requests per second × mean
request time (Little's law) gives 3.6 requests in the system for DeepSeek and 15.9 for GLM, and the request rate
differed by 10%.

### 1. Decode speed per user, by batch-size band

Per-user speed of a decode log line = its `gen throughput` ÷ its `#running-req`. "All" = every decode line in the
band, as the protocol specifies; the protocol names no statistic, and we report the median (means below). "Clean"
keeps only the lines whose interval contained no prefill chunk.

| running requests | DeepSeek: lines | DeepSeek tok/s | GLM v3.6, 16→17 Oct: lines | GLM tok/s | GLM, 15→16 Oct | DeepSeek ÷ GLM |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 53,731 | **510** | 3,938 | 264 | 289 | **1.94×** |
| 2–8 | 117,036 | **346** | 34,980 | 181 | 173 | **1.91×** |
| 9–16 | 3,714 | **169** | 23,932 | 115 | 117 | **1.46×** |
| 17–32 | 54 | (138) | 12,661 | 76 | 80 | too few lines |
| 33–64 | 0 | — | 1,440 | 44 | 54 | no data |

- **The top two bands could not be compared.** DeepSeek never had more than 20 requests running (GLM: 55); 98% of
  its decode lines were at 1–8 requests (GLM: 51%). 54 lines (40 decode steps each) is not a measurement.
- Day to day, the GLM medians moved by 2–9% in bands 1 to 17–32 and by 18% at 33–64. The DeepSeek differences
  (+46% to +94%) are well outside that.
- Clean decode (no prefill in the interval): DeepSeek 560 / 449 / 293 tok/s in bands 1 / 2–8 / 9–16 (40,385 /
  55,502 / 214 lines) against GLM 274 / 215 / 171 / 141 / 116 in bands 1 to 33–64.
- Means instead of medians: DeepSeek 547 / 351 / 177; GLM 280 / 177 / 115 / 78 / 48.
- Context per running request (KV tokens in the batch ÷ requests, median): DeepSeek 68k–87k, GLM 88k–97k.
- Cross-check from a different source, the server's inter-token-latency histogram (every generated token, all
  batch sizes): mean 3.25 ms on DeepSeek, 12.9 ms on GLM (12.1 ms the day before); p99 29 ms vs 223 ms.

### 2. Time to first token

Deltas of the server's TTFT histogram between consecutive 5-minute snapshots, summed over the window; quantiles
interpolated inside the histogram buckets (bucket in brackets, because the buckets are wide).

| | requests | p50 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|
| DeepSeek | 107,279 | **0.39 s** | **1.45 s** (1–2) | **6.0 s** (4–6) | 0.72 s |
| GLM v3.6, 16→17 Oct | 121,234 | 0.99 s | 4.0 s (4–6) | 14.7 s (10–20) | 1.81 s |
| GLM v3.6, 15→16 Oct | 130,274 | 0.90 s | 3.7 s (2–4) | 11.3 s (10–20) | 1.58 s |

By 6-hour block (from the window start), p90 was 1.9 / 1.3 / 1.4 / 0.9 s on DeepSeek and 5.3 / 5.9 / 1.9 / 2.3 s on
GLM. Time in the waiting queue (same method): p50 / p90 / p99 = 0.005 / 0.37 / 4.8 s on DeepSeek, 0.008 / 1.8 / 9.8 s
on GLM. Request time end to end: p50 1.5 s vs 3.9 s, p99 27 s vs 132 s. The DeepSeek window also had less prefill
work, so part of this gap is load, not the server.

### 3. Prefill throughput, full 4,096-token chunks

`input throughput` of the prefill log lines with exactly 4,096 new tokens. In both SGLang builds that number is the
chunk's tokens divided by the time since the previous prefill line, so the decode steps run between two chunks are
inside it: 16 of them on DeepSeek, 2 on GLM. The protocol's number is the first row; the split below it separates
the two effects.

| p50 / p90, tok/s | DeepSeek | GLM v3.6, 16→17 Oct | GLM v3.6, 15→16 Oct |
|---|---|---|---|
| **all full chunks (protocol)** | **14,139 / 20,398** (35,325 chunks) | **17,723 / 19,542** (93,764) | 18,149 / 19,592 (88,947) |
| while requests were decoding | 13,919 / 16,053 (30,278) | 17,720 / 19,533 (93,524) | 18,146 / 19,589 (88,846) |
| back-to-back full chunks, nothing decoding | 24,836 / 25,316 (3,481) | 21,991 / 23,064 (184) | 20,672 / 22,008 (93) |

The prefill itself is about 13% faster on DeepSeek; under traffic each chunk waits for 16 decode steps, so prompts
are admitted about 20% slower than on GLM. That is the first tuning target (section 2: the interval, 16 against
lower).

### 4. Speculation acceptance, by band

Mean of the per-line values. Accept length = tokens kept per verification step, the bonus token included;
accept rate = (length − 1) ÷ draft tokens, so the rates of the two methods are not comparable (5 draft tokens on
DeepSeek, 3 on GLM).

| running requests | DeepSeek DSpark: length / rate | GLM v3.6 EAGLE: length / rate |
|---|---|---|
| 1 | 3.90 / 0.579 | 2.90 / 0.632 |
| 2–8 | 3.71 / 0.541 | 2.83 / 0.610 |
| 9–16 | 3.61 / 0.522 | 2.81 / 0.602 |
| 17–32 | (3.69 / 0.538, 54 lines) | 2.81 / 0.603 |
| 33–64 | — | 2.68 / 0.617 |

### 5. Cache, traffic, concurrency, queueing

| | DeepSeek | GLM v3.6, 16→17 Oct | GLM v3.6, 15→16 Oct |
|---|---:|---:|---:|
| prefix-cache hit (cached ÷ prompt tokens) | **96.7%** | 96.1% | 96.3% |
| requests per hour | 4,564 | 5,063 | 5,458 |
| prompt tokens per hour (of which not cached) | 334 M (11.0 M) | 486 M (19.2 M) | 496 M (18.5 M) |
| generated tokens per hour | 3.03 M | 3.73 M | 4.10 M |
| requests running: max in the logs | 20 | 55 | 54 |
| requests running at the 5-min snapshots, p50 / p90 / max | 2 / 6.6 / 12 | 11 / 27 / 48 | 13 / 25 / 42 |
| waiting queue: max in the logs | 26 | 32 | 30 |
| snapshots with a waiting queue | 20% | 33% | 29% |
| decode lines with a waiting queue | 10.2% | 5.3% | 5.4% |

In all three windows every decode line with a waiting queue also had free slots (64 and 96 were never reached):
requests waited for prefill, not for room.

### 6. Errors

| | DeepSeek | GLM v3.6, 16→17 Oct |
|---|---|---|
| chat completions, server HTTP log | 107,521: 107,514 × 200, 6 × 400, 1 × 401; no 5xx | 120,121: 120,108 × 200, 13 × 401; no 5xx |
| server tracebacks, out-of-memory, CUDA errors, retracted requests | 0 | 0 |
| aborted requests (server counter, mostly client disconnects) | 59 | 214 |
| restarts | 2, both in the gaps above; 0 by the watchdog | 0 |

- The six 400s: five are one client retrying a 1,100,034-token prompt, over the 1,048,576-token context
  (14:44–14:45 UTC); one is an invalid image (15:46 UTC). The server also logged one failed health check at 11:21:07,
  31 s after boot, before warm-up finished.
- The server's HTTP log records the status when the response starts; an error inside a stream after the 200 is not
  visible there.
- **There is no gateway log with HTTP statuses on this server**, so the protocol's "HTTP non-200 at the gateway"
  cannot be reported as written. The closest edge view is the Cloudflare tunnel's failure log: outside the gaps,
  68 failed requests on DeepSeek, all cancelled by the client; inside the two gaps, 354 "connection refused" while
  the server was down. On the GLM day the same log has 238, including 107 "unable to reach the origin" errors
  (connection reset or closed by the server side; not investigated). The tunnel's share of traffic changed during the window (the direct port closed at 18:08 UTC),
  so the counts are not comparable across windows.

### 7. Memory

| | DeepSeek | GLM v3.6, 16→17 Oct |
|---|---|---|
| free GPU memory (nvidia-smi), per GPU | ~38.6 GiB right after a boot; within the first hour of traffic down to a plateau, then flat (±0.02 GiB): 5.0 GiB (11:21–17:23), 6.35 GiB (18:08–11:21); the four GPUs identical | 10.8 / 11.7 / 11.7 / 12.1 GiB (GPU 0–3), flat |
| KV pool | 20.0 M tokens | 8.77 M tokens |
| KV held by running requests, p50 / p90 / max | 180 k / 528 k / 1.57 M tokens (max 7.9% of the pool) | 1.13 M / 2.61 M / 4.55 M (52%) |
| rest of the pool | reusable prefix cache, 19.8 M tokens at the median; 2,304 tokens free; 1.04 B tokens evicted in 23.5 h | prefix cache 7.6 M; 1.62 B evicted in 24 h |
| sliding-window (SWA) pool, 1,655,808 tokens | in use p50 1,536, p90 3,840, max 16,128 tokens (<1%) | (no SWA; KDA state slots used p50 44, max 196 of 700) |

No leak over the 17 hours of the last server process. The pool is full, but of cache: the hit rate is already 96.7%.

### How each number was computed

- **Decode, acceptance, prefill, concurrency in the logs:** TP0 `Decode batch` and `Prefill batch` lines of the
  SGLang logs, outside the gaps. Decode lines with zero throughput are skipped (there were none). "Clean" = the
  previous TP0 scheduler line was also a decode line. Prefill split: "nothing decoding" = `#running-req` 0 on this
  line and on the previous full chunk, which directly precedes it.
- **TTFT, queue time, end-to-end, inter-token latency, cache, requests, tokens, aborts, evictions:** deltas between
  consecutive 5-minute snapshots of `/metrics` from the same server process, dropping pairs that span a restart or
  a gap; histogram quantiles by linear interpolation inside the bucket. The queue-time histogram is reported by
  each of the four ranks, so its count is 4× the requests; the quantiles are unaffected.
- **GPU memory:** `nvidia-smi` total − used in the same snapshots (GiB). **Pool use:** the server's gauges in the
  snapshots and the usage fields of the decode lines.
- **HTTP statuses:** the server's access log (uvicorn), `POST /v1/chat/completions`. Clients are counted by hashed
  address only, to identify our own study traffic; no request content is read.
- `generated tokens` comes from a counter that is updated when a request finishes, so hourly values near the edges
  of a window are approximate.
