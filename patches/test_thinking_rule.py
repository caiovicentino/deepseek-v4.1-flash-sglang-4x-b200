"""Unit check of the patched thinking rule (run with the dsv41 image python, SGLANG_DEFAULT_THINKING=1)."""
from sglang.srt.entrypoints.openai.serving_chat import _cb_thinking_requested as rule


class Req:
    def __init__(self, kw):
        self.chat_template_kwargs = kw


cases = [
    (None, True),
    ({}, True),
    ({"thinking": False}, False),
    ({"thinking": True}, True),
    ({"enable_thinking": False}, False),
    ({"enable_thinking": True}, True),
    ({"thinking": True, "enable_thinking": False}, True),
]
bad = [(k, e) for k, e in cases if rule(Req(k)) != e]
print("thinking rule:", "OK" if not bad else f"FAIL {bad}")
