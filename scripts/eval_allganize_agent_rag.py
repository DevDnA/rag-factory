"""Allganize agent RAG 평가 하니스.

8+ 한국어 평가 쿼리(다양한 도메인/의도)를 /auto SSE 엔드포인트로 호출해
각 쿼리의 (verdict / 인용 매칭 / token count / 종료 이벤트 / 도메인/의도 / 정답 파일)
를 JSON으로 기록한다.

사용:
    uv run python scripts/eval_allganize_agent_rag.py \\
        --queries scripts/eval_allganize_queries.json \\
        --out allganize-eval-project/output/agent_rag_eval_<ts>.json \\
        --label baseline

실패 분류 (계산 후 보고서 'failure_type' 필드):
  - empty_answer: token 이벤트 0건 또는 답변 문자 수 < 10
  - verifier_fail: verification.verdict == "FAIL"
  - citation_miss: warning 이벤트 발생 (환각 인용)
  - wrongful_refusal: 답변에 refusal 문구가 있는데 target_file_name이 sources에 포함
  - clarification_loop: clarification 이벤트가 발생 + token 이벤트 0건
  - http_error: 5xx 또는 transport error
  - timeout: 클라이언트 타임아웃
  - none: 위 어느 것에도 해당 없음
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

REFUSAL_PATTERNS = (
    "해당 정보는 제공된 문서에 포함",
    "관련 정보를 찾을 수 없",
    "확인할 수 없",
    "제공된 문서에서 찾을 수 없",
)


def run_query(client: httpx.Client, base_url: str, query: str, timeout: float) -> dict:
    t_start = time.perf_counter()
    t_first_token: float | None = None
    answer_chunks: list[str] = []
    sources: list[dict] = []
    route_event: dict | None = None
    verification: dict | None = None
    warning: dict | None = None
    clarification: dict | None = None
    actions: list[str] = []
    observations: list[str] = []
    last_event_type: str | None = None
    error: str | None = None

    try:
        with client.stream(
            "POST",
            f"{base_url}/auto",
            json={"query": query},
            timeout=timeout,
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
                last_event_type = etype
                if etype == "token":
                    if t_first_token is None:
                        t_first_token = time.perf_counter() - t_start
                    answer_chunks.append(evt.get("content", ""))
                elif etype == "sources":
                    sources = evt.get("sources", [])
                elif etype == "route":
                    route_event = {"mode": evt.get("mode"), "intent": evt.get("intent")}
                elif etype == "verification":
                    verification = {"verdict": evt.get("verdict"), "issues": evt.get("issues", [])}
                elif etype == "warning":
                    warning = {"content": evt.get("content"), "items": evt.get("items", [])}
                elif etype == "clarification":
                    clarification = {"content": evt.get("content")}
                elif etype == "action":
                    actions.append(json.dumps(evt.get("content"), ensure_ascii=False)[:200])
                elif etype == "observation":
                    observations.append(json.dumps(evt.get("content"), ensure_ascii=False)[:200])
                elif etype == "done":
                    break
    except httpx.TimeoutException as exc:
        error = f"timeout: {exc}"
    except httpx.HTTPStatusError as exc:
        error = f"http_status: {exc.response.status_code}"
    except httpx.HTTPError as exc:
        error = f"http_error: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = f"other: {type(exc).__name__}: {exc}"

    t_total = time.perf_counter() - t_start
    answer = "".join(answer_chunks)
    return {
        "ttft_sec": round(t_first_token, 3) if t_first_token else None,
        "total_sec": round(t_total, 3),
        "n_chars": len(answer),
        "answer": answer,
        "sources": sources,
        "route": route_event,
        "verification": verification,
        "warning": warning,
        "clarification": clarification,
        "actions": actions,
        "observations": observations,
        "last_event_type": last_event_type,
        "error": error,
    }


def classify_failure(q: dict, r: dict) -> str:
    if r["error"]:
        if "timeout" in r["error"]:
            return "timeout"
        return "http_error"
    if r["clarification"] and r["n_chars"] == 0:
        return "clarification_loop"
    if r["n_chars"] < 10:
        return "empty_answer"
    if r["verification"] and r["verification"]["verdict"] == "FAIL":
        return "verifier_fail"
    if r["warning"]:
        return "citation_miss"
    # wrongful refusal: 답변이 corpus profile name을 포함한 OOD-style 거절문이거나
    # 일반 refusal patterns에 매칭되는데, target file이 코퍼스에 존재(또는 검색됨)하는
    # 경우. corpus profile name이 dominant 1 doc으로 편중된 land mine 케이스를 잡기 위함.
    expected_doc = q.get("target_file_name") or q.get("expected_doc")
    answer = r["answer"]
    # Heuristic: "본 시스템은 ... 특화" 패턴 = OOD-style refusal (corpus profile-driven)
    is_ood_refusal = ("본 시스템은" in answer and "특화" in answer)
    is_classic_refusal = any(p in answer for p in REFUSAL_PATTERNS) and r["n_chars"] < 400
    if expected_doc:
        source_doc_names = {s.get("source_doc_id", "") for s in r["sources"]}
        expected_stem = Path(expected_doc).stem
        retrievable = any(expected_stem in (s or "") for s in source_doc_names)
        if (is_ood_refusal or is_classic_refusal) and (retrievable or is_ood_refusal):
            return "wrongful_refusal"
    elif is_ood_refusal:
        # synthetic query labeled refusal/ambiguous — OOD refusal is correct, not wrongful
        pass
    return "none"


def check_expected_doc_in_sources(q: dict, r: dict) -> bool:
    expected_doc = q.get("target_file_name") or q.get("expected_doc")
    if not expected_doc:
        return True  # synthetic queries don't have ground-truth doc
    expected_stem = Path(expected_doc).stem
    # sources have ``source_doc_id`` (file name) not ``doc_id`` (uuid).
    return any(expected_stem in (s.get("source_doc_id") or "") for s in r["sources"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    results = []

    with httpx.Client() as client:
        for i, q in enumerate(queries):
            qid = q.get("id", f"q{i:02d}")
            print(f"[{qid}] intent={q.get('intent')} domain={q.get('domain')}  Q={q['query'][:80]}", flush=True)
            r = run_query(client, args.base_url, q["query"], args.timeout)
            r["query_id"] = qid
            r["query"] = q["query"]
            r["intent"] = q.get("intent")
            r["domain"] = q.get("domain")
            r["target_file_name"] = q.get("target_file_name")
            r["target_answer"] = q.get("target_answer")
            r["failure_type"] = classify_failure(q, r)
            r["expected_doc_in_sources"] = check_expected_doc_in_sources(q, r)
            results.append(r)
            verdict_disp = (r["verification"] or {}).get("verdict", "-")
            print(
                f"  failure={r['failure_type']}  verdict={verdict_disp}  chars={r['n_chars']}  "
                f"sources={len(r['sources'])}  expected_in_sources={r['expected_doc_in_sources']}  "
                f"ttft={r['ttft_sec']}s  total={r['total_sec']}s",
                flush=True,
            )

    # Summary
    n = len(results)
    by_failure: dict[str, int] = {}
    for r in results:
        by_failure[r["failure_type"]] = by_failure.get(r["failure_type"], 0) + 1
    n_failed = sum(v for k, v in by_failure.items() if k != "none")

    summary = {
        "label": args.label,
        "n_queries": n,
        "n_failures": n_failed,
        "failure_breakdown": by_failure,
        "n_expected_doc_in_sources": sum(1 for r in results if r["expected_doc_in_sources"]),
        "avg_ttft_sec": round(
            sum(r["ttft_sec"] or 0 for r in results) / max(n, 1), 3
        ),
        "avg_total_sec": round(sum(r["total_sec"] for r in results) / max(n, 1), 3),
    }
    out = {"label": args.label, "summary": summary, "results": results, "queries": queries}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== [{args.label}] summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
