#!/usr/bin/env python3
"""Ralph Quality Loop 벤치마크 하니스.

slm-factory의 ``/auto`` SSE 엔드포인트를 ``benchmark/queries.json``의 모든
질의로 호출하고, 각 질의에 대해:

- 라우팅 결정(intent / mode)
- Ralph iteration 별 score / reflector / reviewer 통과 상태
- ``<promise>DONE</promise>`` 발행 여부
- 답변 길이 + 참고 문서 수
- 종단 latency

를 수집해 ``benchmark/results/<run_name>.json``에 저장합니다.

사용 예
--------

```
# 서버가 8000에서 떠 있다고 가정
uv run python benchmark/bench.py --threshold 7.5 --run-name t75
uv run python benchmark/bench.py --threshold 7.0 --run-name t70 --override-only
```

``--override-only``를 주면 my/project.yaml을 수정하지 않고 호출 측 헤더(향후
지원)만 사용합니다. 현재는 my/project.yaml의 ralph_loop_quality_threshold를
직접 변경한 뒤 서버를 재시작하는 흐름을 권장합니다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_QUERIES = _PROJECT_ROOT / "benchmark" / "queries.json"
_DEFAULT_RESULTS = _PROJECT_ROOT / "benchmark" / "results"


@dataclass
class IterationRecord:
    iteration: int
    score: float | None
    reflector_ok: bool
    reviewer_passed: bool
    failed_reviewers: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    id: str
    query: str
    expected_intent: str
    elapsed_sec: float
    route_mode: str | None
    detected_intent: str | None
    verbalization: str | None
    ralph_iterations: list[IterationRecord] = field(default_factory=list)
    promise_emitted: bool = False
    final_iteration: int = 0
    last_score: float | None = None
    sources_count: int = 0
    answer_chars: int = 0
    clarification_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ralph_iterations"] = [asdict(it) for it in self.ralph_iterations]
        return d


async def _stream_query(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    *,
    expected_intent: str,
    qid: str,
) -> QueryResult:
    """단일 질의를 /auto에 보내고 SSE를 끝까지 소비해 메트릭을 수집합니다."""
    started = time.perf_counter()
    result = QueryResult(
        id=qid,
        query=query,
        expected_intent=expected_intent,
        elapsed_sec=0.0,
        route_mode=None,
        detected_intent=None,
        verbalization=None,
    )
    answer_chars = 0
    saw_first_thought_iter0 = False

    try:
        async with client.stream(
            "POST",
            f"{base_url}/auto",
            json={"query": query},
            timeout=httpx.Timeout(900.0, connect=10.0),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "route":
                    result.route_mode = event.get("mode")
                    result.detected_intent = event.get("intent")
                elif etype == "thought":
                    # 라우팅 직후의 의도 발화 thought (iteration=0)
                    if (
                        not saw_first_thought_iter0
                        and event.get("iteration") == 0
                    ):
                        result.verbalization = event.get("content", "")
                        saw_first_thought_iter0 = True
                elif etype == "ralph_iteration":
                    rec = IterationRecord(
                        iteration=int(event.get("iteration", 0)),
                        score=event.get("score"),
                        reflector_ok=bool(event.get("reflector_ok", True)),
                        reviewer_passed=bool(event.get("reviewer_passed", True)),
                        failed_reviewers=list(event.get("failed_reviewers") or []),
                    )
                    result.ralph_iterations.append(rec)
                    result.final_iteration = rec.iteration
                    if rec.score is not None:
                        result.last_score = rec.score
                elif etype == "promise":
                    result.promise_emitted = True
                elif etype == "token":
                    answer_chars += len(event.get("content", "") or "")
                elif etype == "sources":
                    result.sources_count = len(event.get("sources") or [])
                elif etype == "clarification":
                    result.clarification_count = len(event.get("questions") or [])
                elif etype == "done":
                    break
                elif etype == "error":
                    result.error = str(event.get("content") or "unknown")
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.answer_chars = answer_chars
    result.elapsed_sec = round(time.perf_counter() - started, 3)
    return result


async def _run_all(
    queries_path: Path,
    base_url: str,
    run_name: str,
    results_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    data = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = data.get("queries") or []
    if not queries:
        raise ValueError(f"질의가 비어 있습니다: {queries_path}")

    results: list[QueryResult] = []
    async with httpx.AsyncClient() as client:
        for q in queries:
            qid = q.get("id") or q.get("query")
            print(f"[bench] ▶ {qid}: {q['query'][:48]}…", flush=True)
            r = await _stream_query(
                client,
                base_url,
                q["query"],
                expected_intent=q.get("intent", ""),
                qid=qid,
            )
            kind = "PROMISE" if r.promise_emitted else (
                f"max@{r.final_iteration}" if r.final_iteration else "no-iter"
            )
            score_str = f"{r.last_score:.1f}" if r.last_score is not None else "—"
            print(
                f"  ✓ {r.elapsed_sec:.1f}s | intent={r.detected_intent or '?'}"
                f" | iter={r.final_iteration} | score={score_str}"
                f" | {kind} | answer={r.answer_chars}자 | sources={r.sources_count}",
                flush=True,
            )
            results.append(r)

    # 요약 통계.
    promise_count = sum(1 for r in results if r.promise_emitted)
    iters = [r.final_iteration for r in results if r.final_iteration]
    scores = [r.last_score for r in results if r.last_score is not None]
    elapsed = [r.elapsed_sec for r in results]
    summary = {
        "run_name": run_name,
        "total_queries": len(results),
        "promise_emitted_count": promise_count,
        "promise_rate": (promise_count / len(results)) if results else 0.0,
        "avg_iterations": (sum(iters) / len(iters)) if iters else 0,
        "avg_last_score": (sum(scores) / len(scores)) if scores else None,
        "avg_elapsed_sec": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
        "p50_elapsed_sec": sorted(elapsed)[len(elapsed) // 2] if elapsed else 0.0,
        "max_elapsed_sec": max(elapsed) if elapsed else 0.0,
        "metadata": metadata,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{run_name}.json"
    out.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [r.to_dict() for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"[bench] 저장: {out}")
    print(
        f"[bench] 요약 — promise={promise_count}/{len(results)}"
        f" ({summary['promise_rate']:.0%}) |"
        f" avg_iter={summary['avg_iterations']:.2f} |"
        f" avg_score={summary['avg_last_score'] or '—'} |"
        f" avg_latency={summary['avg_elapsed_sec']:.1f}s"
    )
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--queries", default=str(_DEFAULT_QUERIES))
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS))
    p.add_argument("--run-name", required=True, help="결과 파일명(확장자 제외)")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="현재 서버에 적용된 ralph_loop_quality_threshold (메타데이터 기록용)",
    )
    p.add_argument(
        "--label",
        default="",
        help="실행 라벨/노트 — 결과 메타데이터에 포함",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = {
        "threshold": args.threshold,
        "base_url": args.base_url,
        "label": args.label,
    }
    asyncio.run(
        _run_all(
            queries_path=Path(args.queries),
            base_url=args.base_url,
            run_name=args.run_name,
            results_dir=Path(args.results_dir),
            metadata=metadata,
        )
    )


if __name__ == "__main__":
    main()
