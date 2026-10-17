# Operations

| script | what |
|---|---|
| `swap.sh` | stop the running SGLang by PID, wait until the GPUs are free, launch `serve/serve.sh`, wait for `/health`, run `ROLLBACK` if it does not come up |

## How the 2026-10-17 swap went

| UTC | |
|---|---|
| 10:41 | checkpoint download started on the production host while GLM kept serving (510 GB in 82 s over the Hub's Xet transfer) |
| 10:43–10:47 | nightly image extracted without Docker into a separate directory (`tools/extract_image.py`, 69 layers, 31 GB) |
| 10:48 | every flag checked against `launch_server --help` of the new build; rollback kit saved on the host and in our ops repo |
| 10:49 | swap started (10 requests running); GLM drained, GPUs free at 10:51 |
| 11:08 | DeepSeek healthy (17 min: weights, Engram host tables, FlashInfer autotune), validation 16/16 |
| 11:12 | reasoning found in `message.content` by reading the outputs; root cause in SGLang, patch written and unit-tested |
| 11:14–11:21 | restart with the patch (7 min, autotune cached); 16/16 + leak checks 5/5; watchdog back |

Total time without service: 24 minutes, in two windows. What we would do differently: run the leak checks before
the first swap — the bug was visible on the first request that did not set `thinking`.

## Restart watchdog

Ours polls `/health` every 30 s; if the server process is gone twice in a row, or `/health` fails for 10 minutes with
the process alive, it restarts — DeepSeek first, the previous GLM recipe as the fallback — with at most 3 restarts in
6 hours, and stays out of the way while a maintenance file exists. Boot timeout is 25 minutes because a cold
DeepSeek boot takes 17. The GLM watchdog is in the
[GLM 4× B200 repo](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200/tree/main/ops).
