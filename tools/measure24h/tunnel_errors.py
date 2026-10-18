#!/usr/bin/env python3
"""Request failures recorded by the Cloudflare tunnel (cloudflared) per measurement window, inside and outside the
declared gaps. There is no gateway log with HTTP statuses on the server; this is the closest thing at the edge.
Addresses and stream numbers are stripped from the error strings. Windows and gaps are the ones in measure24h.py.
Usage: python3 tunnel_errors.py [/root/cloudflared.log] > tunnel_errors.json"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure24h import WINDOWS  # noqa: E402

LOG = sys.argv[1] if len(sys.argv) > 1 else "/root/cloudflared.log"
LINE = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z ERR Request failed error="([^"]*)"')
ADDR = re.compile(r"\[?[0-9a-f:.]+\]?:\d+|\d+\.\d+\.\d+\.\d+")


def norm(e):
    e = re.sub(r"stream \d+", "stream N", e)
    return ADDR.sub("ADDR", e)


out = {}
for k, w in WINDOWS.items():
    t0, t1 = w["t0"].replace(" ", "T"), w["t1"].replace(" ", "T")
    gaps = [(a.replace(" ", "T"), b.replace(" ", "T")) for a, b, _ in w["gaps"]]
    res = {"outside_gaps": {}, "inside_gaps": {}}
    for raw in open(LOG, errors="replace"):
        m = LINE.match(raw)
        if not m or not (t0 <= m.group(1) < t1):
            continue
        where = "inside_gaps" if any(a <= m.group(1) < b for a, b in gaps) else "outside_gaps"
        e = norm(m.group(2))
        res[where][e] = res[where].get(e, 0) + 1
    out[k] = res
print(json.dumps(out, indent=1))
