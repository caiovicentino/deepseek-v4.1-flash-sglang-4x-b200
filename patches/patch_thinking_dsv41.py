"""cbmachine 17/10/2026: make SGLang's DSv4.1 encoder and reasoning parser use ONE thinking rule.

Measured on the nightly-dev-cu13-20261016 image: with SGLANG_DEFAULT_THINKING=1 and a request that does not say
anything about thinking, the encoder renders a thinking prompt (serving_chat._apply_jinja_template reads the env
default) but _get_reasoning_from_request does not force reasoning parsing, so the raw reasoning and "</think>" land in
message.content. GLM-era clients (opencode) also send chat_template_kwargs.enable_thinking, which the encoder ignored.

Rule (encoder and parser): `thinking` if present; else `enable_thinking` (alias); else SGLANG_DEFAULT_THINKING.
Usage: python patch_thinking_dsv41.py /path/to/serving_chat.py   (idempotent check: refuses to patch twice)
"""
import sys

p = sys.argv[1]
s = open(p).read()
old_enc = '''        thinking_requested = (request.chat_template_kwargs or {}).get(
            "thinking", envs.SGLANG_DEFAULT_THINKING.get()
        )'''
new_enc = '''        thinking_requested = _cb_thinking_requested(request)'''
old_par = '''        if not self.reasoning_parser:
            return False

        if self.reasoning_parser == "iquest_q1":'''
new_par = '''        if not self.reasoning_parser:
            return False

        if self.chat_encoding_spec == "dsv41":
            return _cb_thinking_requested(request)

        if self.reasoning_parser == "iquest_q1":'''
helper = '''

def _cb_thinking_requested(request) -> bool:
    """cbmachine 17/10/2026: one thinking rule for the DSv4.1 encoder and the reasoning parser. `thinking` wins;
    GLM-era clients send `enable_thinking`, accepted as an alias; otherwise the server default."""
    kw = request.chat_template_kwargs or {}
    if "thinking" in kw:
        return bool(kw["thinking"])
    if "enable_thinking" in kw:
        return bool(kw["enable_thinking"])
    return bool(envs.SGLANG_DEFAULT_THINKING.get())
'''
assert "_cb_thinking_requested" not in s, "already patched"
assert s.count(old_enc) == 1, "encoder site not found exactly once"
assert s.count(old_par) == 1, "parser site not found exactly once"
s = s.replace(old_enc, new_enc).replace(old_par, new_par)
i = s.index("\nclass OpenAIServingChat")
s = s[:i] + helper + s[i:]
open(p, "w").write(s)
print("patched", p)
