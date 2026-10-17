"""Validation of DeepSeek-V4.1-Flash-1017 before it takes production (2026-10-17). Usage: API_KEY=... python val_dsv41.py
Protocol from memory: kernel boundaries, 3k-token pt-BR generation, greedy code, tools, image, reasoning effort,
client payload shapes (opencode/Hermes), streaming. Prints PASS/FAIL per check and a summary line."""
import ast, base64, json, re, struct, sys, time, urllib.request, zlib
import os
KEY = os.environ["API_KEY"]; URL = os.environ.get("URL", "http://127.0.0.1:8000"); M = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4.1-flash"
RES = []
def call(body, timeout=900, stream=False):
    body = {"model": M, **body}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        if not stream: return json.loads(resp.read())
        chunks, text = 0, ""
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data:") and not line.endswith("[DONE]"):
                d = json.loads(line[5:]); chunks += 1
                for ch in d.get("choices") or []: text += (ch.get("delta") or {}).get("content") or ""
        return chunks, text
def check(name, ok, info=""):
    RES.append((name, ok)); print(("PASS " if ok else "FAIL ") + name + (" :: " + info if info else ""), flush=True)
def msg(d): return d["choices"][0]["message"]
ANOM = re.compile(r"[^\x00-\x7FÀ-ſ\s‐-… -¿]")
# 1. kernel boundaries: a code word at the start, filler after it, question at the end
filler = "O relatorio trimestral descreve vendas, custos e metas da equipe de operacoes. "
for words in [12, 30, 60, 130, 700, 2600, 7000]:
    p = "A palavra-codigo e GIRASSOL-417. Guarde-a.\n" + filler * max(1, words // 12) + "\nQual e a palavra-codigo? Responda so com ela."
    try:
        d = call({"messages": [{"role": "user", "content": p}], "max_tokens": 600, "temperature": 0, "chat_template_kwargs": {"thinking": False}})
        out = (msg(d).get("content") or ""); pt = d["usage"]["prompt_tokens"]
        check(f"fronteira prompt={pt} tok", "GIRASSOL-417" in out.upper().replace(" ", ""), repr(out[:60]))
    except Exception as e: check(f"fronteira {words}", False, repr(e)[:120])
# 2. long pt-BR generation, 3k tokens
d = call({"messages": [{"role": "user", "content": "Escreva um ensaio longo, em portugues do Brasil, sobre a historia da computacao no Brasil, com secoes e muitos detalhes. Pelo menos 2500 palavras."}],
          "max_tokens": 3200, "temperature": 0.7, "chat_template_kwargs": {"thinking": False}})
t = msg(d).get("content") or ""; n = d["usage"]["completion_tokens"]
blocks = [len(ANOM.findall(t[i:i + 2000])) for i in range(0, len(t), 2000)]
check("pt-BR longo", n >= 2500 and max(blocks or [0]) <= 3, f"{n} tokens, anomalias por bloco {blocks}, final: {t[-160:]!r}")
# 3. greedy code
d = call({"messages": [{"role": "user", "content": "Escreva uma funcao Python `mediana(xs)` que devolve a mediana de uma lista de numeros, com docstring e tratamento de lista vazia (ValueError). Responda so com um bloco de codigo."}],
          "max_tokens": 1500, "temperature": 0, "chat_template_kwargs": {"thinking": False}})
c = msg(d).get("content") or ""; m = re.search(r"```(?:python)?\n(.*?)```", c, re.S); src = m.group(1) if m else c
try:
    ns = {}; exec(compile(ast.parse(src), "m", "exec"), ns); f = ns["mediana"]
    ok = f([3, 1, 2]) == 2 and f([4, 1, 3, 2]) == 2.5
    try: f([]); ok = False
    except ValueError: pass
    check("codigo guloso", ok)
except Exception as e: check("codigo guloso", False, repr(e)[:120])
# 4. tool call
tools = [{"type": "function", "function": {"name": "clima", "description": "Clima atual de uma cidade", "parameters": {"type": "object", "properties": {"cidade": {"type": "string"}}, "required": ["cidade"]}}}]
d = call({"messages": [{"role": "user", "content": "Como esta o clima em Recife agora?"}], "tools": tools, "max_tokens": 2000})
tc = msg(d).get("tool_calls") or []
try: ok = bool(tc) and tc[0]["function"]["name"] == "clima" and "recife" in json.loads(tc[0]["function"]["arguments"])["cidade"].lower()
except Exception: ok = False
check("tool call", ok, json.dumps(tc)[:160])
# 4b. tool result round trip with reasoning_content kept
if ok:
    hist = [{"role": "user", "content": "Como esta o clima em Recife agora?"},
            {"role": "assistant", "content": msg(d).get("content") or "", "reasoning_content": msg(d).get("reasoning_content") or "", "tool_calls": tc},
            {"role": "tool", "tool_call_id": tc[0].get("id", "x"), "content": json.dumps({"temp_c": 29, "ceu": "ensolarado"})}]
    d2 = call({"messages": hist, "tools": tools, "max_tokens": 2000})
    check("tool result", "29" in (msg(d2).get("content") or ""), repr((msg(d2).get("content") or "")[:100]))
# 5. image: 64x64 solid red PNG
def png(r, g, b, w=64, h=64):
    raw = b"".join(b"\x00" + bytes([r, g, b]) * w for _ in range(h))
    ch = lambda t, x: struct.pack(">I", len(x)) + t + x + struct.pack(">I", zlib.crc32(t + x) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")
img = "data:image/png;base64," + base64.b64encode(png(220, 20, 20)).decode()
try:
    d = call({"messages": [{"role": "user", "content": [{"type": "text", "text": "Qual a cor desta imagem? Uma palavra."}, {"type": "image_url", "image_url": {"url": img}}]}], "max_tokens": 800})
    out = (msg(d).get("content") or "").lower(); check("imagem", "vermelh" in out or "red" in out, repr(out[:60]))
except Exception as e: check("imagem", False, repr(e)[:160])
# 6. reasoning effort low vs high, and client shapes
lens = {}
for eff in ["low", "high"]:
    d = call({"messages": [{"role": "user", "content": "Quanto e 17*23? Responda so o numero."}], "reasoning_effort": eff, "max_tokens": 4000})
    lens[eff] = len(msg(d).get("reasoning_content") or ""); ok = "391" in (msg(d).get("content") or "")
    check(f"raciocinio {eff}", ok, f"{lens[eff]} chars de raciocinio")
d = call({"messages": [{"role": "user", "content": "Diga ola em uma palavra."}], "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 300})
check("payload opencode enable_thinking=false", bool((msg(d).get("content") or "").strip()), repr((msg(d).get("content") or "")[:60]))
# 7. streaming
chunks, text = call({"messages": [{"role": "user", "content": "Conte de 1 a 20."}], "stream": True, "max_tokens": 800, "chat_template_kwargs": {"thinking": False}}, stream=True)
check("streaming", chunks > 5 and "20" in text, f"{chunks} chunks")
ok_n = sum(1 for _, o in RES if o); print(f"RESUMO {ok_n}/{len(RES)} PASS", flush=True)
