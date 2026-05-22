"""allganize/RAG-Evaluation-Dataset-KO → rag-factory queries.json 변환.

사용:
    uv run python scripts/build_allganize_eval.py \\
        --domains public finance law \\
        --out benchmark/queries.json

스키마 매핑:
    allganize.question        → query
    allganize.target_answer   → expected
    allganize.target_file_name → expected_doc      (retrieval ground-truth)
    allganize.target_page_no   → expected_page     (retrieval ground-truth)
    allganize.context_type     → context_type      (paragraph | table | image)
    allganize.domain           → domain
    (없음)                     → intent = "qa"      (단일 placeholder — IntentClassifier가 query별 자동 분류)
    target_answer 길이         → difficulty (heuristic: <50 easy / 50-200 medium / >200 hard)

bench_streaming.py 호환: id/intent/query 3개 필수 필드 보존, 그 외는 메타데이터로 무시됨.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "allganize/RAG-Evaluation-Dataset-KO"
VALID_DOMAINS = {"finance", "public", "medical", "law", "commerce"}


def difficulty_from_answer(answer: str) -> str:
    n = len(answer)
    if n < 50:
        return "easy"
    if n < 200:
        return "medium"
    return "hard"


def slugify_domain(domain: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", domain.lower())


def convert(domains: list[str]) -> dict:
    ds = load_dataset(DATASET_ID, split="test")
    selected = ds.filter(lambda r: r["domain"] in domains)

    queries: list[dict] = []
    domain_counters: dict[str, int] = {}
    for row in selected:
        domain = row["domain"]
        domain_counters[domain] = domain_counters.get(domain, 0) + 1
        idx = domain_counters[domain]
        q: dict = {
            "id": f"allganize_{slugify_domain(domain)}_{idx:03d}",
            "intent": "qa",
            "query": row["question"],
            "expected": row["target_answer"],
            "expected_doc": row["target_file_name"],
            "expected_page": row["target_page_no"],
            "context_type": row["context_type"],
            "domain": domain,
            "difficulty": difficulty_from_answer(row["target_answer"] or ""),
        }
        queries.append(q)

    return {
        "domain": "allganize RAG-Evaluation-Dataset-KO ({})".format(", ".join(domains)),
        "description": (
            "한국어 RAG 평가 표준 벤치 (allganize/RAG-Evaluation-Dataset-KO, MIT). "
            "도메인 필터: {}. 총 {}개 QA. "
            "retrieval ground-truth(expected_doc + expected_page)와 context_type(paragraph/table/image) "
            "필드 보유 — Recall@k, MRR, modality별 정확도 분리 측정 가능. "
            "intent는 단일 'qa'로 통일(평가셋엔 intent 라벨 없음), 실제 라우팅은 IntentClassifier가 query별로 분류."
        ).format(", ".join(domains), len(queries)),
        "source": {
            "dataset": DATASET_ID,
            "split": "test",
            "license": "MIT",
            "url": f"https://huggingface.co/datasets/{DATASET_ID}",
        },
        "schema": {
            "id": "string — allganize_{domain}_{idx} 안정적 식별자",
            "intent": "string — 모두 'qa' (placeholder, IntentClassifier가 자동 라우팅)",
            "query": "string — 원본 question",
            "expected": "string — target_answer (substring 매칭 평가용)",
            "expected_doc": "string — target_file_name (retrieval GT — 답이 있는 PDF)",
            "expected_page": "string — target_page_no (retrieval GT — 답이 있는 페이지)",
            "context_type": "paragraph | table | image",
            "domain": "finance | public | medical | law | commerce",
            "difficulty": "easy | medium | hard (target_answer 길이 기반 휴리스틱)",
        },
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["public", "finance", "law"],
        help="필터링할 도메인 (allganize 5개 중 선택)",
    )
    parser.add_argument("--out", default="benchmark/queries.json")
    args = parser.parse_args()

    unknown = set(args.domains) - VALID_DOMAINS
    if unknown:
        raise SystemExit(f"unknown domains: {unknown}. valid: {VALID_DOMAINS}")

    payload = convert(args.domains)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[built] {out_path} — {len(payload['queries'])} queries")
    from collections import Counter
    by_domain = Counter(q["domain"] for q in payload["queries"])
    by_difficulty = Counter(q["difficulty"] for q in payload["queries"])
    by_context = Counter(q["context_type"] for q in payload["queries"])
    print(f"  domain:     {dict(by_domain)}")
    print(f"  difficulty: {dict(by_difficulty)}")
    print(f"  context:    {dict(by_context)}")


if __name__ == "__main__":
    main()
