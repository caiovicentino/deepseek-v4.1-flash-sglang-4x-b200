#!/usr/bin/env python3
"""Builds results_24h.json (the numbers in docs/results.md section 3; no client identifiers) from the server-side
outputs of measure24h.py and tunnel_errors.py. Runs locally, no server access.
Usage: python3 summarize.py RAW.json TUNNEL.json OUT.json [INTERNAL.json]
INTERNAL.json = {"client-<hash>": "why we think it is our own traffic"}; it stays local, because a SHA-256 of an IPv4
address can be brute-forced. The output names those clients internal-1, internal-2, ..."""
import json
import sys

raw = json.load(open(sys.argv[1]))
tunnel = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
INTERNAL = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
W = ["dsv41", "glm_v36", "glm_v36_prev"]
BANDS = ["1", "2-8", "9-16", "17-32", "33-64"]


def r1(x, d=1):
    return None if x is None else round(x, d)


def bucket_bounds(h, q):
    """The histogram bucket (lower, upper] that holds quantile q."""
    bs = sorted(((float(k), v) for k, v in h["buckets"].items()), key=lambda kv: kv[0])
    total = bs[-1][1]
    prev = 0.0
    for b, c in bs:
        if c >= q * total:
            return [prev, b]
        prev = b
    return None


out = {"protocol": "docs/results.md section 2 (written 2026-10-17, before the data)",
       "generated_from": raw["generated_utc"], "windows": raw["windows"]}

dec = {}
for b in BANDS:
    row = {}
    for w in W:
        d = raw["logs"][w]["decode"][b]
        row[w] = {"lines": d["lines"], "share_of_decode_lines": d["share_of_decode_lines"],
                  "per_user_p50_all": d["per_user_all"].get("p50"), "per_user_mean_all": d["per_user_all"].get("mean"),
                  "per_user_p10_all": d["per_user_all"].get("p10"), "per_user_p90_all": d["per_user_all"].get("p90"),
                  "lines_clean": d["lines_clean"], "per_user_p50_clean": d["per_user_clean"].get("p50"),
                  "kv_tokens_per_running_req_p50": d["kv_tokens_per_running_req"].get("p50")}
    a, g, g2 = (row[w]["per_user_p50_all"] for w in W)
    row["ratio_dsv41_over_glm_v36_p50_all"] = r1(a / g, 2) if a and g else None
    row["glm_day_to_day_change_p50_all"] = r1(g / g2 - 1, 3) if g and g2 else None
    dec[b] = row
out["1_decode_per_user_tok_s"] = dec

out["2_ttft_s"] = {w: {**{k: raw["series"][w]["ttft_s"][k] for k in ("n", "mean", "p50", "p90", "p99")},
                       "p90_bucket": bucket_bounds(raw["series"][w]["ttft_s"], 0.9),
                       "p99_bucket": bucket_bounds(raw["series"][w]["ttft_s"], 0.99),
                       "six_hour_blocks": {i: v["ttft_s"] for i, v in raw["series"][w]["six_hour_blocks"].items()}}
                   for w in W}
out["2b_queue_time_s"] = {w: {k: raw["series"][w]["queue_time_s"][k] for k in ("n", "mean", "p50", "p90", "p99")}
                          for w in W}
out["2c_e2e_latency_s"] = {w: {k: raw["series"][w]["e2e_latency_s"][k] for k in ("n", "mean", "p50", "p90", "p99")}
                           for w in W}
out["2d_inter_token_latency_s"] = {w: {k: raw["series"][w]["inter_token_latency_s"][k]
                                       for k in ("n", "mean", "p50", "p90", "p99")} for w in W}

out["3_prefill_full_chunks_tok_s"] = {
    w: {"protocol_all_full_chunks": raw["logs"][w]["prefill"]["throughput_full_chunks"]["all_full"],
        "with_requests_decoding": raw["logs"][w]["prefill"]["throughput_full_chunks"]["full_busy"],
        "no_request_decoding": raw["logs"][w]["prefill"]["throughput_full_chunks"]["full_idle"],
        "pure_prefill_consecutive_full_chunks_nothing_decoding":
            raw["logs"][w]["prefill"]["throughput_full_chunks"]["full_after_full_idle"],
        "prefill_lines": raw["logs"][w]["prefill"]["lines"], "new_tokens": raw["logs"][w]["prefill"]["new_tokens"]}
    for w in W}

out["4_speculation"] = {b: {w: {"accept_len_mean": raw["logs"][w]["decode"][b]["accept_len"].get("mean"),
                                "accept_rate_mean": raw["logs"][w]["decode"][b]["accept_rate"].get("mean"),
                                "lines": raw["logs"][w]["decode"][b]["lines"]} for w in W} for b in BANDS}

out["5_traffic_cache_concurrency"] = {
    w: {**raw["series"][w]["totals"],
        "running_req_max_in_logs": raw["logs"][w]["running_req_over_decode_lines"]["max"],
        "running_req_at_5min_snapshots": raw["series"][w]["gauges_at_snapshots"]["num_running_reqs"],
        "queue_req_max_in_logs": max(raw["logs"][w]["queue_req_over_decode_lines"]["max"],
                                     raw["logs"][w]["prefill"]["queue_max"]),
        "snapshots_with_queue": raw["series"][w]["snapshots_with_queue"], "snapshots": raw["series"][w]["snapshots"],
        "decode_lines_with_queue_and_free_slots": raw["logs"][w]["decode_lines_with_queue_and_free_slots"],
        "decode_lines": raw["logs"][w]["running_req_over_decode_lines"]["n"],
        "littles_law_mean_in_system": r1(raw["series"][w]["totals"]["requests_per_hour"] / 3600
                                         * raw["series"][w]["e2e_latency_s"]["mean"], 2)}
    for w in W}

errs = {}
for w in W:
    lg = raw["logs"][w]
    chat = lg["http"]["chat_by_status"]
    errs[w] = {"chat_completions_by_status": chat,
               "chat_non200_events": [[t, st] for t, st, _ in lg["http"]["chat_non200_events"]],
               "non200_other_paths": {k: v for k, v in lg["http"]["non200_all_paths"].items()
                                      if "/v1/chat/completions" not in k},
               "log_categories": lg["errors"]["categories"], "exception_classes": lg["errors"]["exception_classes"],
               "events": lg["errors"]["events"],
               "aborted_requests_counter": raw["series"][w]["totals"]["aborted_requests"],
               "retracted_max_at_snapshots": raw["series"][w]["gauges_at_snapshots"]["num_retracted_reqs"]["max"],
               "tunnel_request_failures": tunnel[w]}
out["6_errors"] = errs

out["7_memory"] = {w: {"gpu_free_gib": raw["series"][w]["gpu_free_gib"],
                       "kv": {k: raw["series"][w]["gauges_at_snapshots"][k] for k in
                              ("kv_used_tokens", "kv_evictable_tokens", "kv_available_tokens", "token_usage")},
                       "evicted_tokens": raw["series"][w]["totals"]["evicted_tokens"],
                       "pools_in_decode_lines": raw["logs"][w]["pool_usage_over_decode_lines"]}
                   for w in W}
out["7_memory"]["dsv41"]["gpu_free_plateau_gib"] = {
    "segment_2026-10-17_11:21-17:23": 5.01, "segment_17:40-18:03": 8.40, "segment_18:08-11:21": 6.35,
    "note": "gpu_free_timeline.py: ~38.6 GiB right after boot, plateau reached within the first hour of traffic, "
            "flat to 0.02 GiB afterwards; identical on the four GPUs"}

nchat = raw["logs"]["dsv41"]["http"]["chat_clients"]["total"]
per = raw["logs"]["dsv41"]["http"]["chat_clients"]["per_hour"]
internal = {f"internal-{i + 1}": {"requests": sum(per.get(c, {}).values()), "per_hour": per.get(c, {}), "why": why}
            for i, (c, why) in enumerate(INTERNAL.items())}
out["internal_load"] = {"dsv41_window": internal,
                        "total_requests": sum(v["requests"] for v in internal.values()),
                        "share_of_chat_requests": round(sum(v["requests"] for v in internal.values()) / nchat, 4),
                        "glm_windows": "none of these addresses (one request on 2026-10-15) and not the RTX bench "
                                       "address; unknown addresses cannot be ruled out"}
out["boot_facts_dsv41"] = {
    "weights_per_gpu_gib": 74.50, "dspark_head_per_gpu_gib": 2.14,
    "pools_per_gpu_gib": round(99.97 - 47.94, 2), "cuda_graphs_per_gpu_gib": round(5.21 + 0.29, 2),
    "available_after_allocation_gib": [41.03, 41.14, 41.22, 41.22],
    "kv_pool_tokens": {"10:51 boot": 20019200, "11:16 boot": 20019200, "17:36 boot": 19998976,
                       "18:04 boot": 19998976},
    "boot_launch_to_ready": {"10:51:31 first (cold)": "16 min 46 s", "11:16:12": "4 min 24 s",
                             "17:36:34": "3 min 27 s", "18:04:54": "3 min 20 s"},
    "units": "SGLang logs 'GB' computed as bytes / 2**30 (GiB); total per GPU 183,359 MiB = 179.1 GiB"}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print("ok", OUT)
