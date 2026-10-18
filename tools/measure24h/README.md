# 24-hour production measurement

The scripts behind section 3 of [`docs/results.md`](../../docs/results.md). They only read: the SGLang logs, the
5-minute series of `/metrics` + `nvidia-smi` snapshots, and the Cloudflare tunnel log. Windows, gaps, log files and
server process IDs are set at the top of `measure24h.py`; they are ours, change them for your host.

| script | where | what |
|---|---|---|
| `measure24h.py OUT.json` | server | everything in the protocol: decode by batch band, prefill, acceptance, concurrency and HTTP statuses from the logs; TTFT, queue time, cache, traffic and memory from the series |
| `tunnel_errors.py [cloudflared.log]` | server | request failures at the tunnel per window, inside and outside the gaps |
| `clients_after.py LOG FROM TO` | server | hashed client addresses in a time range (used once to recognise our own study traffic) |
| `gpu_free_timeline.py FROM TO` | server | free GPU memory per hour and server process |
| `summarize.py RAW.json TUNNEL.json OUT.json [INTERNAL.json]` | anywhere | the published numbers ([`docs/data/results_24h.json`](../../docs/data/results_24h.json)) |

Client addresses are only ever hashed, and the hashes stay local: a SHA-256 of an IPv4 address can be reversed by
trying them all. So `measure24h.py`'s raw output, `known_clients.json` (addresses to label) and `INTERNAL.json`
(hashes we recognised as our own traffic) are not in this repository. No request content reaches any output.
