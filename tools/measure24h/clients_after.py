#!/usr/bin/env python3
"""Hashed client ids (same hash as measure24h.py) seen on POST /v1/chat/completions in a time range of one log.
Used once to tell which hashed client is our own study harness (it started at 2026-10-18 11:31 UTC, after the window).
Usage: python3 clients_after.py LOG 'YYYY-MM-DD HH:MM:SS' 'YYYY-MM-DD HH:MM:SS'"""
import hashlib, re, sys
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] INFO:\s+(\S+?):\d+ - \"POST /v1/chat/completions HTTP/[\d.]+\" (\d{3})")
cnt = {}
for line in open(sys.argv[1], errors="replace"):
    m = TS.match(line)
    if m and sys.argv[2] <= m.group(1) < sys.argv[3]:
        c = "client-" + hashlib.sha256(m.group(2).encode()).hexdigest()[:8]
        first, last, n = cnt.get(c, (m.group(1), m.group(1), 0))
        cnt[c] = (first, m.group(1), n + 1)
for c, (a, b, n) in sorted(cnt.items(), key=lambda kv: -kv[1][2]):
    print(c, n, a, b)
