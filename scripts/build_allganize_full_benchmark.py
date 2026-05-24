"""allganize 공식 ground-truth에서 대규모 벤치마크 query를 추출한다.

LLM 생성 금지 — allganize ground-truth(question/target_answer/target_file_name/
context_type/domain)를 그대로 추출만 한다. intent는 query 텍스트 휴리스틱으로
부여(compare/lookup/explain/howto). corpus 32개 unique 문서 중 allganize에
매칭되는 문서를 도메인/의도/context_type 다양성을 고려해 균형 추출한다.

사용:
    uv run python scripts/build_allganize_full_benchmark.py \\
        --target 50 --out scripts/eval_allganize_queries_full.json

OOD/ambiguous 2개는 기존 10-query 세트의 q09/q10을 그대로 차용해
(retrieval-miss 외) OOD gate / clarification 모드도 벤치마크에 포함시킨다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

ARROW = (
    "/Users/devdna/.cache/huggingface/datasets/"
    "allganize___rag-evaluation-dataset-ko/default/0.0.0/"
    "35db50ff8739d78b11ee68b6f4ef04a95862a504/"
    "rag-evaluation-dataset-ko-test.arrow"
)
DOCDIR = Path("allganize-eval-project/documents")


def normalize_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[\s_\-\[\]()+]+", "", s)
    return s.lower()


def load_allganize() -> list[dict]:
    with pa.memory_map(ARROW, "r") as src:
        try:
            rb = ipc.open_stream(src)
        except Exception:  # noqa: BLE001
            src.seek(0)
            rb = ipc.open_file(src)
        tbl = rb.read_all()
    cols = ["domain", "question", "target_answer", "target_file_name",
            "target_page_no", "context_type"]
    return tbl.select(cols).to_pylist()


def corpus_unique_files() -> list[str]:
    seen: dict[str, str] = {}
    for f in sorted(DOCDIR.glob("*.pdf")):
        h = hashlib.md5(f.read_bytes()).hexdigest()
        seen.setdefault(h, f.name)
    return sorted(seen.values())


def match_allganize_to_corpus(ag_files: set[str], corpus_files: list[str]) -> dict[str, str]:
    """allganize target_file_name -> corpus stem 매핑(정규화 stem)."""
    corpus_stems = {Path(n).stem: n for n in corpus_files}
    corpus_norm = {normalize_stem(n): Path(n).stem for n in corpus_files}
    matched: dict[str, str] = {}
    for agf in sorted(ag_files):
        ags = Path(agf).stem
        if ags in corpus_stems:
            matched[agf] = ags
            continue
        n = normalize_stem(agf)
        if n in corpus_norm:
            matched[agf] = corpus_norm[n]
            continue
        hit = None
        for cn, cs in corpus_norm.items():
            if n and (n in cn or cn in n) and abs(len(n) - len(cn)) <= 8:
                hit = cs
                break
        if hit:
            matched[agf] = hit
    return matched


def classify_intent(q: str) -> str:
    """query 텍스트 휴리스틱 intent 분류."""
    ql = q.lower()
    if any(k in q for k in ("차이", "비교", "다른 점", "차이점", "vs", "대비", "각각")):
        return "compare"
    if any(k in q for k in ("어떻게", "방법", "절차", "어떤 순서", "하려면", "방안")):
        return "howto"
    if any(k in q for k in ("무엇", "어떤 서류", "얼마", "몇", "언제", "어디", "누구", "어느")):
        return "lookup"
    return "explain"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_allganize()
    corpus_files = corpus_unique_files()
    ag_files = {r["target_file_name"] for r in rows}
    matched = match_allganize_to_corpus(ag_files, corpus_files)

    matchable = [r for r in rows if r["target_file_name"] in matched]
    # group by corpus stem so we can guarantee per-doc coverage
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in matchable:
        by_doc[matched[r["target_file_name"]]].append(r)

    # ---- round-robin pick: 1 per covered doc first (coverage guarantee),
    #      then fill by domain balance up to target ----
    picked: list[dict] = []
    used_questions: set[str] = set()

    def take(r: dict, corpus_stem: str) -> None:
        used_questions.add(r["question"])
        picked.append({
            "domain": r["domain"],
            "context_type": r["context_type"],
            "question": r["question"],
            "target_answer": r["target_answer"],
            "target_file_name": r["target_file_name"],
            "corpus_stem": corpus_stem,
        })

    # pass 1: one query per covered corpus doc (prefer paragraph/table over image
    # since image-context queries are unanswerable without OCR — still include some)
    for stem, items in sorted(by_doc.items()):
        order = sorted(items, key=lambda x: {"paragraph": 0, "table": 1, "text": 2, "image": 3}.get(x["context_type"], 9))
        take(order[0], stem)

    # pass 2: fill to target with domain balance, avoid dup questions, vary context_type
    remaining = [(r, matched[r["target_file_name"]]) for r in matchable
                 if r["question"] not in used_questions]
    # sort to interleave domains and prefer table/image (harder, more diagnostic value)
    dom_counts = Counter(p["domain"] for p in picked)
    remaining.sort(key=lambda rc: (dom_counts[rc[0]["domain"]],
                                   {"table": 0, "image": 1, "paragraph": 2, "text": 3}.get(rc[0]["context_type"], 9)))
    for r, stem in remaining:
        if len(picked) >= args.target - 2:  # reserve 2 slots for OOD+ambiguous
            break
        if r["question"] in used_questions:
            continue
        take(r, stem)
        dom_counts[r["domain"]] += 1

    # finalize ids with intent
    out_queries: list[dict] = []
    for i, p in enumerate(picked):
        intent = classify_intent(p["question"])
        qid = f"ag{i:02d}_{intent}_{p['domain']}_{p['context_type']}"
        out_queries.append({
            "id": qid,
            "intent": intent,
            "domain": p["domain"],
            "context_type": p["context_type"],
            "query": p["question"],
            "target_file_name": p["target_file_name"],
            "target_answer": p["target_answer"],
        })

    # append OOD + ambiguous control queries (mode coverage: OOD gate, clarification)
    out_queries.append({
        "id": "ag_ctrl_refusal_ood",
        "intent": "refusal",
        "domain": "out_of_domain",
        "context_type": "control",
        "query": "2024 파리 올림픽 한국 양궁팀의 금메달 개수는 몇 개인가요?",
        "target_file_name": None,
        "target_answer": "[OUT OF CORPUS — should refuse]",
    })
    out_queries.append({
        "id": "ag_ctrl_ambiguous",
        "intent": "ambiguous",
        "domain": "finance",
        "context_type": "control",
        "query": "그건 어떻게 되나요?",
        "target_file_name": None,
        "target_answer": "[AMBIGUOUS — should clarify]",
    })

    Path(args.out).write_text(json.dumps(out_queries, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- coverage report to stdout ----
    covered_corpus = set(matched.values())
    corpus_stems = [Path(n).stem for n in corpus_files]
    uncovered = [s for s in corpus_stems if s not in covered_corpus]
    picked_stems = {p["corpus_stem"] for p in picked}

    print(f"wrote {len(out_queries)} queries -> {args.out}")
    print(f"  ground-truth queries: {len(picked)}  + control: 2")
    print(f"  intents: {Counter(q['intent'] for q in out_queries)}")
    print(f"  domains: {Counter(q['domain'] for q in out_queries)}")
    print(f"  context_types: {Counter(q['context_type'] for q in out_queries)}")
    print(f"\n=== COVERAGE: corpus unique docs={len(corpus_files)} ===")
    print(f"  allganize-matchable corpus docs: {len(covered_corpus)}")
    print(f"  covered by >=1 benchmark query:  {len(picked_stems)}")
    print(f"\n  | corpus doc stem | #bench queries |")
    print(f"  |---|---|")
    per_doc = Counter(p["corpus_stem"] for p in picked)
    for stem in sorted(corpus_stems):
        n = per_doc.get(stem, 0)
        mark = "" if stem in covered_corpus else " (no allganize query)"
        print(f"  | {stem[:50]} | {n}{mark} |")
    print(f"\n  UNCOVERED corpus docs ({len(uncovered)}): reason = allganize has no query for them")
    for u in uncovered:
        print(f"    - {u}")


if __name__ == "__main__":
    main()
