"""LLM-judge 정답률 평가.

eval 하니스가 만든 baseline JSON의 각 생성답변을 allganize target_answer와
비교해 qwen3.5:9b(Ollama, temperature 0, seed 고정, think 비활성)로
PASS / PARTIAL / FAIL + 한 줄 사유를 판정한다.

OOD/ambiguous(=target_answer가 [OUT OF CORPUS / [AMBIGUOUS) 항목은 정답률
판정 대상이 아니므로 verdict="SKIP"으로 표기하고 카운트에서 제외한다.

eval 하니스와 동일한 9b 모델을 쓰므로 반드시 eval 종료 후 순차 실행
(동시 호출 금지 — 24GB 통합 메모리 swap thrashing).

사용:
    uv run python scripts/judge_eval_answers.py \\
        --eval allganize-eval-project/output/agent_rag_eval_<ts>_full_baseline.json \\
        --out  allganize-eval-project/output/agent_rag_eval_<ts>_full_baseline_judged.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx

JUDGE_MODEL = "qwen3.5:9b"
OLLAMA = "http://localhost:11434"

PROMPT = """당신은 한국어 RAG 답변을 채점하는 엄정한 평가자입니다.
아래 [질문]에 대한 [모범답안]과 [채점대상답변]을 비교해 사실 정합성을 판정하세요.

판정 기준:
- PASS: 모범답안의 핵심 사실(수치/주체/결론)을 정확히 담고 모순·환각이 없음.
- PARTIAL: 일부 핵심 사실은 맞으나 누락·불완전하거나 일부 부정확.
- FAIL: 핵심 사실이 틀렸거나, 답을 못 했거나(거절/무응답), 환각이 있음.

반드시 아래 형식 한 줄로만 답하세요(설명 금지):
VERDICT: <PASS|PARTIAL|FAIL> | <한 줄 사유>

[질문]
{question}

[모범답안]
{target_answer}

[채점대상답변]
{answer}
"""

VERDICT_RE = re.compile(r"VERDICT\s*:\s*(PASS|PARTIAL|FAIL)\s*\|?\s*(.*)", re.IGNORECASE)


def is_control(target_answer: str | None) -> bool:
    if not target_answer:
        return True
    return target_answer.strip().startswith("[OUT OF CORPUS") or target_answer.strip().startswith("[AMBIGUOUS")


def judge_one(client: httpx.Client, question: str, target_answer: str, answer: str) -> dict:
    prompt = PROMPT.format(question=question, target_answer=target_answer, answer=answer or "(빈 응답)")
    resp = client.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": 256},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    text = (resp.json().get("response") or "").strip()
    m = VERDICT_RE.search(text)
    if m:
        verdict = m.group(1).upper()
        reason = m.group(2).strip()[:300]
    else:
        verdict = "FAIL"
        reason = f"judge 파싱 실패: {text[:120]!r}"
    return {"verdict": verdict, "reason": reason, "raw": text[:500]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    results = data["results"]

    judged = []
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "SKIP": 0}
    with httpx.Client() as client:
        for i, r in enumerate(results):
            qid = r.get("query_id", f"q{i:02d}")
            ta = r.get("target_answer")
            if is_control(ta):
                jr = {"verdict": "SKIP", "reason": "OOD/ambiguous control — 정답률 대상 아님", "raw": ""}
                counts["SKIP"] += 1
                print(f"[{qid}] SKIP (control)", flush=True)
            else:
                jr = judge_one(client, r["query"], ta, r.get("answer", ""))
                counts[jr["verdict"]] = counts.get(jr["verdict"], 0) + 1
                print(f"[{qid}] {jr['verdict']} | {jr['reason'][:90]}", flush=True)
            judged.append({
                "query_id": qid,
                "intent": r.get("intent"),
                "domain": r.get("domain"),
                "context_type": next((q.get("context_type") for q in data.get("queries", []) if q.get("id") == qid), None),
                "query": r["query"],
                "target_file_name": r.get("target_file_name"),
                "target_answer": ta,
                "answer": r.get("answer", ""),
                "expected_doc_in_sources": r.get("expected_doc_in_sources"),
                "failure_type": r.get("failure_type"),
                "verdict": jr["verdict"],
                "reason": jr["reason"],
            })

    scored = counts["PASS"] + counts["PARTIAL"] + counts["FAIL"]
    summary = {
        "eval_file": args.eval,
        "judge_model": JUDGE_MODEL,
        "n_total": len(results),
        "n_scored": scored,
        "n_skipped_control": counts["SKIP"],
        "counts": counts,
        "pass_rate_pct": round(100 * counts["PASS"] / scored, 1) if scored else 0.0,
        "pass_or_partial_rate_pct": round(100 * (counts["PASS"] + counts["PARTIAL"]) / scored, 1) if scored else 0.0,
        "fail_rate_pct": round(100 * counts["FAIL"] / scored, 1) if scored else 0.0,
    }
    out = {"summary": summary, "judged": judged}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== judge summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nJudged: {args.out}")


if __name__ == "__main__":
    main()
