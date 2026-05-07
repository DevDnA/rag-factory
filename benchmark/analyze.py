#!/usr/bin/env python3
"""benchmark/results/*.json 결과를 비교해서 최적 threshold를 추천합니다.

평가 기준
---------
1. **promise_rate**: 통과(전 게이트 통과 + 임계점수)한 질의 비율 — 높을수록 좋음.
2. **avg_iterations**: 평균 반복 횟수 — 낮을수록 latency·비용에 유리.
3. **avg_elapsed_sec**: 평균 응답 시간.
4. **avg_last_score**: scorer 평균 점수.

종합 점수
--------
``composite = promise_rate * 100 - avg_iterations * 5 - avg_elapsed_sec * 0.1``

(가중치는 사용자 환경에 맞춰 ``--w-promise / --w-iter / --w-latency`` 로 조정 가능)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def _load_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "summary" not in data:
            continue
        data["__path"] = str(p)
        runs.append(data)
    return runs


def _composite(summary: dict[str, Any], wp: float, wi: float, wl: float) -> float:
    promise = float(summary.get("promise_rate") or 0.0) * 100.0
    iters = float(summary.get("avg_iterations") or 0.0)
    latency = float(summary.get("avg_elapsed_sec") or 0.0)
    return promise * wp - iters * wi - latency * wl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS))
    p.add_argument("--w-promise", type=float, default=1.0)
    p.add_argument("--w-iter", type=float, default=5.0)
    p.add_argument("--w-latency", type=float, default=0.1)
    args = p.parse_args()

    runs = _load_runs(Path(args.results_dir))
    if not runs:
        print("[analyze] 결과 없음 —", args.results_dir)
        return

    rows: list[tuple[str, dict[str, Any], float]] = []
    for run in runs:
        summary = run["summary"]
        comp = _composite(summary, args.w_promise, args.w_iter, args.w_latency)
        rows.append((run.get("__path", "?"), summary, comp))

    # 비교 테이블.
    print(
        f"{'run':<22} {'thr':>5} {'promise':>9} {'iter':>5}"
        f" {'score':>6} {'avg_s':>7} {'p50':>6} {'max':>6} {'composite':>10}"
    )
    print("-" * 92)
    for path, summary, comp in sorted(rows, key=lambda r: -r[2]):
        run_name = summary.get("run_name", Path(path).stem)
        threshold = (summary.get("metadata") or {}).get("threshold")
        thr_str = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "?"
        promise_pct = float(summary.get("promise_rate") or 0.0) * 100
        avg_iter = float(summary.get("avg_iterations") or 0.0)
        avg_score = summary.get("avg_last_score")
        score_str = f"{avg_score:.2f}" if avg_score else "—"
        avg_s = float(summary.get("avg_elapsed_sec") or 0.0)
        p50 = float(summary.get("p50_elapsed_sec") or 0.0)
        mx = float(summary.get("max_elapsed_sec") or 0.0)
        print(
            f"{run_name:<22} {thr_str:>5} {promise_pct:>7.0f}% "
            f"{avg_iter:>5.2f} {score_str:>6} "
            f"{avg_s:>6.1f}s {p50:>5.1f}s {mx:>5.1f}s {comp:>10.2f}"
        )

    best = max(rows, key=lambda r: r[2])
    best_summary = best[1]
    best_thr = (best_summary.get("metadata") or {}).get("threshold")
    print()
    print(f"[analyze] 추천 threshold: {best_thr}")
    print(f"          run: {best_summary.get('run_name')}")
    print(f"          composite: {best[2]:.2f}")


if __name__ == "__main__":
    main()
