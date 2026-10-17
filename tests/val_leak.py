"""Leak checks added after the 17/10 finding: reasoning must never appear in message.content."""
import json, urllib.request
import os
KEY = os.environ["API_KEY"]; URL = os.environ.get("URL", "http://127.0.0.1:8000")
def call(body, stream=False):
    body = {"model": "deepseek-v4.1-flash", **body}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(r, timeout=600) as resp:
        if not stream:
            m = json.loads(resp.read())["choices"][0]["message"]; return m.get("reasoning_content") or "", m.get("content") or ""
        rc = c = ""
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data:") and not line.endswith("[DONE]"):
                for ch in json.loads(line[5:]).get("choices") or []:
                    d = ch.get("delta") or {}; rc += d.get("reasoning_content") or ""; c += d.get("content") or ""
        return rc, c
Q = [{"role": "user", "content": "Diga ola em uma palavra."}]
ok_all = True
for name, body, want_reasoning in [
    ("padrao", {"messages": Q, "max_tokens": 4000}, True),
    ("padrao stream", {"messages": Q, "max_tokens": 4000, "stream": True}, True),
    ("enable_thinking=false", {"messages": Q, "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4000}, False),
    ("thinking=false", {"messages": Q, "chat_template_kwargs": {"thinking": False}, "max_tokens": 4000}, False),
    ("effort low", {"messages": Q, "reasoning_effort": "low", "max_tokens": 4000}, True),
]:
    rc, c = call(body, stream=body.get("stream", False))
    ok = "</think>" not in c and "<think>" not in c and len(c.strip()) <= 40 and (bool(rc) == want_reasoning)
    ok_all &= ok
    print(("PASS " if ok else "FAIL ") + f"{name:24s} reasoning={len(rc):4d} chars | content={c[:70]!r}", flush=True)
print("LEAK CHECK", "PASS" if ok_all else "FAIL")
