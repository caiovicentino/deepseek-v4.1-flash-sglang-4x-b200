#!/usr/bin/env python3
"""Free GPU memory (nvidia-smi total - used, GiB) per hour and server process from /root/attestation/series.jsonl.
Usage: python3 gpu_free_timeline.py 'YYYY-MM-DDTHH' 'YYYY-MM-DDTHH'"""
import json, sys
rows = {}
for raw in open("/root/attestation/series.jsonl"):
    try:
        d = json.loads(raw)
    except ValueError:
        continue
    if not (sys.argv[1] <= d.get("ts", "") < sys.argv[2]) or not d.get("gpu"):
        continue
    key = (d.get("pid"), d["ts"][:13])
    free = min((g[1] - g[0]) / 1024 for g in d["gpu"])
    r = rows.setdefault(key, [free, free, 0, d["g"].get("num_running_reqs") if d.get("g") else None])
    r[0], r[1], r[2] = min(r[0], free), max(r[1], free), r[2] + 1
for (pid, h), (lo, hi, n, _) in sorted(rows.items(), key=lambda kv: kv[0][1]):
    print(f"{h} pid {pid} snapshots {n} free_min_gib {lo:.2f} free_max_gib {hi:.2f}")
