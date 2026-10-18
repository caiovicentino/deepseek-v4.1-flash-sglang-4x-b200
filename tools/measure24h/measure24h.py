#!/usr/bin/env python3
"""24-hour production measurement on 4x B200: DeepSeek-V4.1-Flash-1017 vs GLM-5.3-Flash v3.6.

Implements the protocol in docs/results.md section 2 (written 2026-10-17, before the data). Read-only: it parses the
SGLang logs and /root/attestation/series.jsonl (one snapshot of /metrics + nvidia-smi every 5 minutes) and writes one
JSON file. Request content never reaches the output: access-log lines give method, path, status and a hashed client
address; error lines are reduced to fixed categories and exception class names.

Usage: python3 measure24h.py OUT.json

Definitions (identical for both models, same code on both logs):
- decode, per user = "gen throughput (token/s)" / "#running-req" of each TP0 "Decode batch" line (one line per 40
  decode steps, --decode-log-interval 40 on both servers). Bands by #running-req: 1, 2-8, 9-16, 17-32, 33-64.
  "all" = every decode line in the band (the protocol's number); "clean" = the previous TP0 scheduler line was also a
  Decode line, i.e. no prefill chunk was logged inside the interval. Lines with gen throughput 0 are skipped.
- prefill = "input throughput (token/s)" of TP0 "Prefill batch" lines with exactly 4,096 new tokens (the chunk size on
  both servers). In both SGLang builds that value is this batch's new tokens divided by the time since the previous
  prefill log line, so the decode steps the scheduler runs between chunks (--prefill-decode-interval: 16 on DeepSeek,
  2 on GLM) are inside the denominator. Reported also split by whether requests were decoding (#running-req > 0).
- TTFT and queue time = percentiles of the deltas of the server's cumulative histograms between consecutive 5-minute
  snapshots of the same server process, summed over the window, linear interpolation inside a bucket (the
  Prometheus histogram_quantile method). Deltas that span a restart or a declared gap are dropped.
- cache hit = delta cached_tokens_total / delta prompt_tokens_total over the same snapshot pairs.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta

BANDS = [("1", 1, 1), ("2-8", 2, 8), ("9-16", 9, 16), ("17-32", 17, 32), ("33-64", 33, 64), (">64", 65, 10_000)]
BANDS_OLD = [("1", 1, 1), ("2-4", 2, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17-32", 17, 32), ("33-64", 33, 64)]
CHUNK = 4096

WINDOWS = {
    "dsv41": {
        "label": "DeepSeek-V4.1-Flash-1017",
        "t0": "2026-10-17 11:21:00", "t1": "2026-10-18 11:21:00",
        "logs": ["/root/dsv41.log.pre-20261017-173443", "/root/dsv41.log.pre-20261017-180303", "/root/dsv41.log"],
        "gaps": [
            ["2026-10-17 17:23:48", "2026-10-17 17:40:01",
             "container restarted by an external SIGTERM; the boot script started an old GLM build, which served "
             "17:30-17:36 UTC (excluded); DeepSeek ready again 17:40:01"],
            ["2026-10-17 18:03:03", "2026-10-17 18:08:14",
             "planned restart: HTTP bound to 127.0.0.1 behind the tunnel, log filter"],
        ],
        "pids": [2605212, 5046, 11222],
        "max_running": 64,
    },
    "glm_v36": {
        "label": "GLM-5.3-Flash v3.6 (most recent full 24 h before the swap)",
        "t0": "2026-10-16 10:49:35", "t1": "2026-10-17 10:49:35",
        "logs": ["/root/glm.log.1", "/root/glm.log.pre-dsv41-20261017-104935"],
        "gaps": [],
        "pids": [2388313],
        "max_running": 96,
    },
    "glm_v36_prev": {
        "label": "GLM-5.3-Flash v3.6 (the 24 h before that, for day-to-day variation)",
        "t0": "2026-10-15 10:49:35", "t1": "2026-10-16 10:49:35",
        "logs": ["/root/glm.log.1"],
        "gaps": [],
        "pids": [2388313],
        "max_running": 96,
    },
}
SERIES = "/root/attestation/series.jsonl"
# Addresses to label instead of hashing; ours stay out of the code, in known_clients.json next to this script
# ({"address": "label"}, optional).
KNOWN_CLIENTS = {"127.0.0.1": "localhost (the Cloudflare tunnel and local tools)"}
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_clients.json")) as _fh:
        KNOWN_CLIENTS.update(json.load(_fh))
except FileNotFoundError:
    pass

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)( TP(\d+)[^\]]*)?\] ?(.*)")
BATCH = re.compile(r"^(Decode|Prefill) batch")
NUM = {k: re.compile(re.escape(k) + r": ([\d.]+)") for k in [
    "#running-req", "#queue-req", "gen throughput (token/s)", "accept len", "accept rate", "#new-seq", "#new-token",
    "#cached-token", "input throughput (token/s)", "full token usage", "swa token usage", "mamba usage",
    "#full token", "#swa token", "#pending-token"]}
ACCESS = re.compile(r'INFO:\s+(\S+?):\d+ - "(\w+) (\S+) HTTP/[\d.]+" (\d{3})')
ERR = [
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("out_of_memory", re.compile(r"OutOfMemoryError|out of memory", re.I)),
    ("cuda_error", re.compile(r"CUDA error|cudaError|CUDA_ERROR")),
    ("sigterm_received", re.compile(r"SIGTERM received")),
    ("sigquit_received", re.compile(r"SIGQUIT received")),
    ("subprocess_crashed", re.compile(r"Subprocess \S+ \(pid=\d+\) crashed")),
    ("health_check_failed", re.compile(r"Health check failed")),
    ("prompt_over_context_warning", re.compile(r"Token indices sequence length is longer than the specified maximum")),
    ("invalid_image_input", re.compile(r"invalid IMAGE data|Error while loading data ImageData")),
    ("error_response_path_warning", re.compile(r"create_error_response")),
    ("retract", re.compile(r"[Rr]etract")),
    ("server_fired_up", re.compile(r"The server is fired up")),
]
EXC = re.compile(r"\b([A-Z]\w*(?:Error|Exception))\b")
LEVEL = re.compile(r"\b(ERROR|WARNING|CRITICAL)\b")


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def num(line, key):
    m = NUM[key].search(line)
    return float(m.group(1)) if m else None


def band_of(n, bands=BANDS):
    for name, lo, hi in bands:
        if lo <= n <= hi:
            return name
    return None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(xs, digits=2):
    if not xs:
        return {"n": 0}
    r = {"n": len(xs), "mean": sum(xs) / len(xs)}
    for q in (0.1, 0.5, 0.9, 0.99):
        r[f"p{int(q * 100)}"] = pct(xs, q)
    r["max"] = max(xs)
    r["min"] = min(xs)
    return {k: (round(v, digits) if isinstance(v, float) else v) for k, v in r.items()}


def hclient(ip):
    return KNOWN_CLIENTS.get(ip) or "client-" + hashlib.sha256(ip.encode()).hexdigest()[:8]


class LogStats:
    def __init__(self, w):
        self.w = w
        self.t0, self.t1 = parse_ts(w["t0"]), parse_ts(w["t1"])
        self.gaps = [(parse_ts(a), parse_ts(b)) for a, b, _ in w["gaps"]]
        self.dec = {b: {"all": [], "clean": [], "acc_len": [], "acc_rate": [], "ctx": []} for b, _, _ in BANDS}
        self.dec_old = {b: {"all": [], "clean": []} for b, _, _ in BANDS_OLD}
        self.dec_zero = 0
        self.running, self.queue = [], []
        self.queue_pos, self.queue_with_room = 0, 0
        self.pools = {"full token usage": [], "swa token usage": [], "mamba usage": [], "#full token": [],
                      "#swa token": []}
        self.pf = {"all_full": [], "full_idle": [], "full_busy": [], "full_after_full_idle": []}
        self.pf_lines, self.pf_new, self.pf_cached, self.pf_full_n = 0, 0, 0, 0
        self.pf_queue_max, self.pf_running_max, self.pf_over_chunk = 0, 0, 0
        self.access = {}
        self.chat_by_client = {}
        self.chat_by_client_hour = {}
        self.non200_events = []
        self.err = {k: 0 for k, _ in ERR}
        self.exc = {}
        self.levels = {}
        self.events = []
        self.hourly = {}
        self.first_ts = self.last_ts = None
        self.reset_file()

    def reset_file(self):
        self.prev_kind = None
        self.prev_pf = None  # (new_tokens, running) of the previous prefill line

    def inside(self, t):
        if not (self.t0 <= t < self.t1):
            return False
        return not any(a <= t < b for a, b in self.gaps)

    def hour(self, t):
        h = t.strftime("%Y-%m-%dT%H")
        return self.hourly.setdefault(h, {"decode_lines": 0, "prefill_full_chunks": 0, "prefill_new_tokens": 0,
                                          "chat_requests": 0, "chat_non200": 0, "running_max": 0,
                                          "queue_max": 0})

    def batch(self, t, tp, kind, body):
        if tp != "0":
            return
        hr = self.hour(t)
        if kind == "Decode":
            n = int(num(body, "#running-req"))
            g = num(body, "gen throughput (token/s)")
            q = int(num(body, "#queue-req") or 0)
            self.running.append(n)
            self.queue.append(q)
            hr["decode_lines"] += 1
            hr["running_max"] = max(hr["running_max"], n)
            hr["queue_max"] = max(hr["queue_max"], q)
            if q > 0:
                self.queue_pos += 1
                if n < self.w["max_running"]:
                    self.queue_with_room += 1
            for k in self.pools:
                v = num(body, k)
                if v is not None:
                    self.pools[k].append(v)
            if g is None or g <= 0 or n <= 0:
                self.dec_zero += 1
            else:
                per = g / n
                clean = self.prev_kind == "Decode"
                b = band_of(n)
                d = self.dec[b]
                d["all"].append(per)
                if clean:
                    d["clean"].append(per)
                a, r = num(body, "accept len"), num(body, "accept rate")
                ft = num(body, "#full token")
                if ft is not None:
                    d["ctx"].append(ft / n)
                if a is not None:
                    d["acc_len"].append(a)
                if r is not None:
                    d["acc_rate"].append(r)
                bo = band_of(n, BANDS_OLD)
                if bo:
                    self.dec_old[bo]["all"].append(per)
                    if clean:
                        self.dec_old[bo]["clean"].append(per)
            self.prev_kind = "Decode"
        else:
            new = int(num(body, "#new-token"))
            cached = int(num(body, "#cached-token") or 0)
            run = int(num(body, "#running-req") or 0)
            q = int(num(body, "#queue-req") or 0)
            thr = num(body, "input throughput (token/s)")
            self.pf_lines += 1
            self.pf_new += new
            self.pf_cached += cached
            self.pf_queue_max = max(self.pf_queue_max, q)
            self.pf_running_max = max(self.pf_running_max, run)
            hr["prefill_new_tokens"] += new
            hr["queue_max"] = max(hr["queue_max"], q)
            if new > CHUNK:
                self.pf_over_chunk += 1
            if new == CHUNK and thr and thr > 0:
                self.pf_full_n += 1
                hr["prefill_full_chunks"] += 1
                self.pf["all_full"].append(thr)
                (self.pf["full_idle"] if run == 0 else self.pf["full_busy"]).append(thr)
                if run == 0 and self.prev_kind == "Prefill" and self.prev_pf == (CHUNK, 0):
                    self.pf["full_after_full_idle"].append(thr)
            self.prev_pf = (new, run)
            self.prev_kind = "Prefill"

    def other(self, t, body):
        m = ACCESS.search(body)
        if m:
            ip, method, path, status = m.groups()
            path = path.split("?")[0]
            key = f"{method} {path} {status}"
            self.access[key] = self.access.get(key, 0) + 1
            if path == "/v1/chat/completions":
                hr = self.hour(t)
                hr["chat_requests"] += 1
                if status != "200":
                    hr["chat_non200"] += 1
                c = hclient(ip)
                self.chat_by_client[c] = self.chat_by_client.get(c, 0) + 1
                ch = self.chat_by_client_hour.setdefault(c, {})
                hk = t.strftime("%Y-%m-%dT%H")
                ch[hk] = ch.get(hk, 0) + 1
                if status != "200":
                    self.non200_events.append([t.strftime("%Y-%m-%d %H:%M:%S"), status, c])
            return
        for k, rx in ERR:
            if rx.search(body):
                self.err[k] += 1
                if k in ("sigterm_received", "server_fired_up", "subprocess_crashed", "out_of_memory",
                         "health_check_failed"):
                    self.events.append([t.strftime("%Y-%m-%d %H:%M:%S"), k])
        for e in set(EXC.findall(body)):
            self.exc[e] = self.exc.get(e, 0) + 1
        lv = LEVEL.search(body)
        if lv:
            self.levels[lv.group(1)] = self.levels.get(lv.group(1), 0) + 1

    def line(self, t, tp, body):
        if self.first_ts is None:
            self.first_ts = t
        self.last_ts = t
        m = BATCH.match(body)
        if m:
            self.batch(t, tp, m.group(1), body)
        else:
            self.other(t, body)

    def result(self):
        dec = {}
        for b, _, _ in BANDS:
            d = self.dec[b]
            dec[b] = {"lines": len(d["all"]), "lines_clean": len(d["clean"]),
                      "per_user_all": summary(d["all"], 1), "per_user_clean": summary(d["clean"], 1),
                      "accept_len": summary(d["acc_len"]), "accept_rate": summary(d["acc_rate"], 3),
                      "kv_tokens_per_running_req": summary(d["ctx"], 0)}
        total = sum(len(self.dec[b]["all"]) for b in self.dec)
        for b in dec:
            dec[b]["share_of_decode_lines"] = round(len(self.dec[b]["all"]) / total, 4) if total else None
        old = {b: {"lines": len(v["all"]), "p50_all": pct(v["all"], 0.5), "p50_clean": pct(v["clean"], 0.5)}
               for b, v in self.dec_old.items()}
        chat = {k: v for k, v in self.access.items() if " /v1/chat/completions " in k}
        non200 = {k: v for k, v in self.access.items() if not k.endswith(" 200")}
        top = sorted(self.chat_by_client.items(), key=lambda kv: -kv[1])
        nchat = sum(self.chat_by_client.values())
        return {
            "log_coverage": {"first": str(self.first_ts), "last": str(self.last_ts)},
            "decode": dec, "decode_reference_bands": old, "decode_lines_skipped_zero_throughput": self.dec_zero,
            "running_req_over_decode_lines": summary(self.running, 1),
            "queue_req_over_decode_lines": summary(self.queue, 1),
            "decode_lines_with_queue": self.queue_pos,
            "decode_lines_with_queue_and_free_slots": self.queue_with_room,
            "pool_usage_over_decode_lines": {k: summary(v, 3) for k, v in self.pools.items() if v},
            "prefill": {"lines": self.pf_lines, "new_tokens": self.pf_new, "cached_tokens_logged": self.pf_cached,
                        "full_chunks": self.pf_full_n, "lines_over_chunk": self.pf_over_chunk,
                        "queue_max": self.pf_queue_max, "running_max": self.pf_running_max,
                        "throughput_full_chunks": {k: summary(v, 0) for k, v in self.pf.items()}},
            "http": {"chat_by_status": chat, "non200_all_paths": non200,
                     "chat_clients": {"total": nchat, "distinct": len(top),
                                      "top": [[c, n, round(n / nchat, 4)] for c, n in top[:8]],
                                      "per_hour": {c: dict(sorted(h.items())) for c, h in self.chat_by_client_hour.items()}},
                     "chat_non200_events": self.non200_events},
            "errors": {"categories": self.err, "exception_classes": self.exc, "log_levels": self.levels,
                       "events": self.events},
            "hourly": dict(sorted(self.hourly.items())),
        }


def scan_logs():
    stats = {k: LogStats(w) for k, w in WINDOWS.items()}
    files = []
    for k, w in WINDOWS.items():
        for f in w["logs"]:
            if f not in files:
                files.append(f)
    for f in files:
        users = [stats[k] for k, w in WINDOWS.items() if f in w["logs"]]
        for s in users:
            s.reset_file()
        last_t = None
        with open(f, errors="replace") as fh:
            for raw in fh:
                m = TS.match(raw)
                if m:
                    last_t = parse_ts(m.group(1))
                    tp, body = m.group(3), m.group(4)
                else:
                    tp, body = None, raw
                if last_t is None:
                    continue
                for s in users:
                    if s.inside(last_t):
                        s.line(last_t, tp, body)
    return {k: s.result() for k, s in stats.items()}


def hist_quantile(buckets, q):
    """buckets: list of (upper_bound, cumulative_count) sorted by bound; +Inf last."""
    total = buckets[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_b, prev_c = 0.0, 0.0
    for b, c in buckets:
        if c >= rank:
            if b == float("inf"):
                return f">{prev_b:g}"
            if c == prev_c:
                return b
            return round(prev_b + (b - prev_b) * (rank - prev_c) / (c - prev_c), 3)
        prev_b, prev_c = b, c
    return None


def hist_summary(delta):
    if not delta:
        return None
    bs = sorted(((float(k), v) for k, v in delta["b"].items()), key=lambda kv: kv[0])
    out = {"n": int(delta["n"]), "mean": round(delta["s"] / delta["n"], 5) if delta["n"] else None}
    for q in (0.5, 0.9, 0.99):
        out[f"p{int(q * 100)}"] = hist_quantile(bs, q)
    out["buckets"] = {f"{b:g}": int(c) for b, c in bs}
    return out


def add_hist(acc, a, b):
    if a is None or b is None:
        return
    acc.setdefault("b", {})
    for k, v in b["b"].items():
        acc["b"][k] = acc["b"].get(k, 0) + v - a["b"].get(k, 0)
    acc["s"] = acc.get("s", 0) + b["s"] - a["s"]
    acc["n"] = acc.get("n", 0) + b["n"] - a["n"]


def scan_series():
    snaps = []
    with open(SERIES) as fh:
        for raw in fh:
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            if d.get("ts", "") >= "2026-10-15":
                snaps.append(d)
    snaps.sort(key=lambda d: d["ts"])
    out = {}
    for k, w in WINDOWS.items():
        t0, t1 = parse_ts(w["t0"]), parse_ts(w["t1"])
        gaps = [(parse_ts(a), parse_ts(b)) for a, b, _ in w["gaps"]]
        pids = set(w["pids"])

        def ok(d):
            t = datetime.strptime(d["ts"], "%Y-%m-%dT%H:%M:%SZ")
            return d.get("pid") in pids and t0 <= t < t1 and not any(a <= t < b for a, b in gaps)

        sel = [d for d in snaps if ok(d)]
        cnt = {}
        hist = {}
        hours = 0.0
        resets = dropped = 0
        hourly = {}
        blocks = {}
        long_pairs = 0
        for a, b in zip(sel, sel[1:]):
            ta = datetime.strptime(a["ts"], "%Y-%m-%dT%H:%M:%SZ")
            tb = datetime.strptime(b["ts"], "%Y-%m-%dT%H:%M:%SZ")
            if a["pid"] != b["pid"] or any(ta < g1 and tb > g0 for g0, g1 in gaps):
                dropped += 1
                continue
            if tb - ta > timedelta(minutes=15):
                long_pairs += 1
            ca, cb = a.get("c") or {}, b.get("c") or {}
            if any(cb.get(x, 0) < ca.get(x, 0) for x in cb):
                resets += 1
                continue
            hours += (tb - ta).total_seconds() / 3600
            dd = {x: cb[x] - ca.get(x, 0) for x in cb}
            for x, v in dd.items():
                cnt[x] = cnt.get(x, 0) + v
            for h in ("time_to_first_token_seconds", "queue_time_seconds", "e2e_request_latency_seconds",
                      "inter_token_latency_seconds"):
                add_hist(hist.setdefault(h, {}), (a.get("h") or {}).get(h), (b.get("h") or {}).get(h))
            hk = tb.strftime("%Y-%m-%dT%H")
            hh = hourly.setdefault(hk, {"hours": 0.0, "c": {}, "ttft": {}})
            hh["hours"] += (tb - ta).total_seconds() / 3600
            for x, v in dd.items():
                hh["c"][x] = hh["c"].get(x, 0) + v
            add_hist(hh["ttft"], (a.get("h") or {}).get("time_to_first_token_seconds"),
                     (b.get("h") or {}).get("time_to_first_token_seconds"))
            bi = int((tb - t0).total_seconds() // (6 * 3600))
            bb = blocks.setdefault(bi, {"hours": 0.0, "c": {}, "ttft": {}, "queue": {}})
            bb["hours"] += (tb - ta).total_seconds() / 3600
            for x, v in dd.items():
                bb["c"][x] = bb["c"].get(x, 0) + v
            add_hist(bb["ttft"], (a.get("h") or {}).get("time_to_first_token_seconds"),
                     (b.get("h") or {}).get("time_to_first_token_seconds"))
            add_hist(bb["queue"], (a.get("h") or {}).get("queue_time_seconds"),
                     (b.get("h") or {}).get("queue_time_seconds"))

        def rates(c, hrs):
            p, kch = c.get("prompt_tokens_total", 0), c.get("cached_tokens_total", 0)
            return {"hours": round(hrs, 3), "requests": int(c.get("num_requests_total", 0)),
                    "requests_per_hour": round(c.get("num_requests_total", 0) / hrs, 1) if hrs else None,
                    "prompt_tokens": int(p), "cached_tokens": int(kch), "uncached_prompt_tokens": int(p - kch),
                    "generation_tokens": int(c.get("generation_tokens_total", 0)),
                    "prompt_tokens_per_hour": round(p / hrs) if hrs else None,
                    "uncached_tokens_per_hour": round((p - kch) / hrs) if hrs else None,
                    "generation_tokens_per_hour": round(c.get("generation_tokens_total", 0) / hrs) if hrs else None,
                    "cache_hit": round(kch / p, 4) if p else None,
                    "aborted_requests": int(c.get("num_aborted_requests_total", 0)),
                    "evicted_tokens": int(c["evicted_tokens_total"]) if "evicted_tokens_total" in c else None}

        g = {x: [] for x in ("num_running_reqs", "num_queue_reqs", "token_usage", "kv_used_tokens",
                             "kv_evictable_tokens", "kv_available_tokens", "num_retracted_reqs", "spec_accept_length",
                             "mamba_used_tokens", "mamba_evictable_tokens")}
        free = [[] for _ in range(4)]
        free_min = [None] * 4
        for d in sel:
            for x in g:
                v = (d.get("g") or {}).get(x)
                if v is not None:
                    g[x].append(v)
            for i, row in enumerate(d.get("gpu") or []):
                f = (row[1] - row[0]) / 1024.0
                free[i].append(f)
                if free_min[i] is None or f < free_min[i][0]:
                    free_min[i] = (round(f, 2), d["ts"])
        out[k] = {
            "snapshots": len(sel), "pairs_dropped_restart_or_gap": dropped, "pairs_counter_reset": resets,
            "pairs_longer_than_15min": long_pairs,
            "totals": rates(cnt, hours),
            "ttft_s": hist_summary(hist.get("time_to_first_token_seconds")),
            "queue_time_s": hist_summary(hist.get("queue_time_seconds")),
            "e2e_latency_s": hist_summary(hist.get("e2e_request_latency_seconds")),
            "inter_token_latency_s": hist_summary(hist.get("inter_token_latency_seconds")),
            "gauges_at_snapshots": {x: summary(v, 3) for x, v in g.items() if v},
            "snapshots_with_queue": sum(1 for v in g["num_queue_reqs"] if v > 0),
            "gpu_free_gib": [dict(summary(f, 2), min_at=free_min[i][1] if free_min[i] else None)
                             for i, f in enumerate(free)],
            "six_hour_blocks": {str(i): {**rates(v["c"], v["hours"]), "ttft_s": {
                q: x for q, x in (hist_summary(v["ttft"]) or {}).items() if q != "buckets"},
                "queue_time_s": {q: x for q, x in (hist_summary(v["queue"]) or {}).items() if q != "buckets"}}
                for i, v in sorted(blocks.items())},
            "hourly": {h: {**rates(v["c"], v["hours"]),
                           "ttft_p50": (hist_summary(v["ttft"]) or {}).get("p50"),
                           "ttft_p90": (hist_summary(v["ttft"]) or {}).get("p90"),
                           "ttft_p99": (hist_summary(v["ttft"]) or {}).get("p99")}
                       for h, v in sorted(hourly.items())},
        }
    return out


def main():
    out = {"generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "windows": {k: {kk: vv for kk, vv in w.items() if kk != "logs"} | {"logs": w["logs"]}
                       for k, w in WINDOWS.items()}}
    out["series"] = scan_series()
    out["logs"] = scan_logs()
    with open(sys.argv[1], "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("ok", sys.argv[1])


if __name__ == "__main__":
    main()
