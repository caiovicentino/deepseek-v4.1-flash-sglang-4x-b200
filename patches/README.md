# Patches over `lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0`

| file | what | why |
|---|---|---|
| `patch_thinking_dsv41.py` | one thinking rule for the V4.1 encoder and the reasoning parser; `enable_thinking` accepted as an alias of `thinking` | with `SGLANG_DEFAULT_THINKING=1`, requests that said nothing about thinking got raw reasoning and `</think>` in `message.content` |
| `test_thinking_rule.py` | unit check of the rule (run with the image's Python, `SGLANG_DEFAULT_THINKING=1`) | 7 cases |
| `apply_encoding_fix.sh` | `encoding_dsv41.py` from upstream commit a038a0dca (#44751) | tool-result image order and current-turn thinking; merged after the nightly |

Both are idempotent-checked: the thinking patch refuses to apply twice; the encoding fix only applies over the exact
parent file (git blob `c04d8059…`). Re-run `tests/val_leak.py` after any image update.
