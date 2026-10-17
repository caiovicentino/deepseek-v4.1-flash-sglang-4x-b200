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

Boot facts: KV pool 20,019,200 tokens; 41 GB free per GPU after allocation; weights ~136 GB per GPU; Engram tables in
host RAM; first boot 17 min, warm boot 7 min.

## 2. 24-hour production measurement — protocol (written before the data, 2026-10-17)

Window: 24 hours from 2026-10-17 11:21 UTC (the patched server's start), on real community traffic.
Comparison: GLM-5.3-Flash v3.6 on the same machine over the same hours of the previous days available in our logs.
Nothing is changed on the server during the window, except a restart the watchdog may need (reported if it happens).

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

To be published on 2026-10-18.
