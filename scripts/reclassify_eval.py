"""기존 eval JSON 보고서를 새 classify_failure 로직으로 재분류합니다.

사용:
    uv run python scripts/reclassify_eval.py <input.json> <output.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse classifier
sys.path.insert(0, str(Path(__file__).parent))
from eval_allganize_agent_rag import classify_failure, check_expected_doc_in_sources  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: reclassify_eval.py <input> <output>")
        sys.exit(2)
    inp, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(inp.read_text(encoding="utf-8"))
    queries_by_id = {q["id"]: q for q in data.get("queries", [])}
    for r in data["results"]:
        q = queries_by_id.get(r["query_id"], {})
        r["failure_type"] = classify_failure(q, r)
        r["expected_doc_in_sources"] = check_expected_doc_in_sources(q, r)
    n = len(data["results"])
    by = {}
    for r in data["results"]:
        by[r["failure_type"]] = by.get(r["failure_type"], 0) + 1
    n_fail = sum(v for k, v in by.items() if k != "none")
    data["summary"]["failure_breakdown"] = by
    data["summary"]["n_failures"] = n_fail
    data["summary"]["n_expected_doc_in_sources"] = sum(
        1 for r in data["results"] if r["expected_doc_in_sources"]
    )
    data["summary"]["label"] = data["summary"].get("label", data.get("label")) + "_reclassified"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"label={data['summary']['label']}  n={n}  n_fail={n_fail}  breakdown={by}")
    print(f"expected_in_sources={data['summary']['n_expected_doc_in_sources']}/{n}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
