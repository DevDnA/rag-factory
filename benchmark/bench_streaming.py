"""스트리밍 latency 벤치 — TTFT, total time, throughput.

사용:
    python benchmark/bench_streaming.py --label verifier_on  --out benchmark/results/streaming_verifier_on.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def run_query(client: httpx.Client, base_url: str, query: str) -> dict:
    t_start = time.perf_counter()
    t_first_token: float | None = None
    n_chars = 0
    n_token_events = 0
    last_evt_type = None

    with client.stream(
        "POST",
        f"{base_url}/auto",
        json={"query": query},
        timeout=300.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")
            last_evt_type = etype
            if etype == "token":
                if t_first_token is None:
                    t_first_token = time.perf_counter() - t_start
                content = evt.get("content", "")
                n_chars += len(content)
                n_token_events += 1
            elif etype == "done":
                break

    t_total = time.perf_counter() - t_start
    gen_time = (t_total - t_first_token) if t_first_token else 0
    return {
        "ttft_sec": round(t_first_token, 3) if t_first_token else None,
        "total_sec": round(t_total, 3),
        "n_chars": n_chars,
        "n_token_events": n_token_events,
        "chars_per_sec": round(n_chars / max(gen_time, 0.001), 1) if t_first_token else 0,
        "last_evt_type": last_evt_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--queries", default="benchmark/queries.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup", action="store_true", help="첫 질의를 warmup으로 분리 측정")
    args = parser.parse_args()

    payload = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    queries = payload["queries"]
    results = []

    with httpx.Client() as client:
        if args.warmup and queries:
            print(f"[warmup] {queries[0]['query'][:50]}...", flush=True)
            w = run_query(client, args.base_url, queries[0]["query"])
            print(
                f"  warmup TTFT={w['ttft_sec']}s total={w['total_sec']}s",
                flush=True,
            )

        for q in queries:
            print(f"[{q['id']}] {q['query'][:50]}...", flush=True)
            r = run_query(client, args.base_url, q["query"])
            r.update({"query_id": q["id"], "intent": q["intent"]})
            results.append(r)
            print(
                f"  TTFT={r['ttft_sec']}s  total={r['total_sec']}s  "
                f"chars={r['n_chars']}  events={r['n_token_events']}  "
                f"throughput={r['chars_per_sec']}c/s",
                flush=True,
            )

    avg_ttft = sum(r["ttft_sec"] or 0 for r in results) / len(results)
    avg_total = sum(r["total_sec"] for r in results) / len(results)
    avg_throughput = sum(r["chars_per_sec"] for r in results) / len(results)
    summary = {
        "label": args.label,
        "n": len(results),
        "avg_ttft_sec": round(avg_ttft, 3),
        "avg_total_sec": round(avg_total, 3),
        "avg_chars_per_sec": round(avg_throughput, 1),
    }

    out = {"label": args.label, "summary": summary, "results": results}
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n[{args.label}] avg TTFT={avg_ttft:.2f}s  "
        f"avg total={avg_total:.2f}s  avg throughput={avg_throughput:.1f}c/s"
    )


if __name__ == "__main__":
    main()
