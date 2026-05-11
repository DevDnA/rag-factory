"""검색 품질 향상용 query enhancement 유틸리티.

RAG 정확도를 높이는 일반적으로 검증된 두 기법을 구현합니다.

1. **HyDE (Hypothetical Document Embeddings)** — Gao et al., 2022.
   LLM이 질의에 대한 이상적 답변을 먼저 생성하고, 그 텍스트를 검색 쿼리로
   사용. 짧은 질의-긴 문서 임베딩 mismatch를 우회해 recall을 끌어올립니다.

2. **Multi-Query Expansion** — LangChain의 `MultiQueryRetriever` 패턴.
   동일 질의를 LLM이 N개 패러프레이즈로 변형 → 각각 검색 → Reciprocal
   Rank Fusion(RRF)으로 결과 병합. 단어 선택 변동성에 강건한 검색.

설계 원칙
---------
- never raise: LLM 호출 실패 시 원본 질의로 fallback (검색은 계속).
- httpx.AsyncClient를 외부에서 주입 (서버 lifespan 객체 재사용).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from ..utils import get_logger

logger = get_logger("rag.query_enhancer")


_HYDE_PROMPT = """당신은 한국어 도메인 전문가입니다. 다음 질의에 대해 **이상적인 검색 결과 문서가
어떤 내용일지를 1~3문단(최대 400자)** 으로 생성하세요. 사용자에게 답하는 게 아니라,
검색 시스템이 의미상 유사 문서를 찾도록 도와주는 가상 문서를 작성합니다.

규칙:
- 도메인 전문어·약어를 자연스럽게 사용 (예: SLA, MIMO, BIS).
- 사실을 단정하지 말고, 일반적인 설명·정의·맥락을 평이하게 기술.
- 답변 형식 금지 — "~입니다", "~할 수 있습니다" 같은 진술문으로만.
- "검색을 위해 이런 내용이 필요합니다" 같은 메타 발언 금지.
- 결과는 평문 텍스트만, Markdown·인용·번호 매기기 모두 금지.

질의: {query}

가상 문서:"""

_MULTI_QUERY_PROMPT = """당신은 한국어 검색 쿼리 변형 전문가입니다. 다음 질의를 의미는 보존하면서
**서로 다른 단어·표현**을 사용한 {n}개의 검색 쿼리로 변형하세요. 도메인 약어 풀어쓰기,
동의어 치환, 어순 변경, 키워드 재배치 등을 자유롭게 사용.

규칙:
- 각 변형은 짧고 명확하게 (15~40자 권장).
- 원본의 핵심 의미·범위는 유지.
- 변형끼리도 서로 표현이 달라야 함 (단순 어미 변경 금지).
- 도메인 약어가 있으면 일부 변형에서 풀어쓰기 ("AP" → "Access Point" / "무선 공유기").

원본 질의: {query}

반드시 다음 JSON 형식으로만 답하세요 (다른 텍스트 금지):
{{"queries": ["변형 1", "변형 2", ...]}}
"""


@dataclass(frozen=True)
class _RankedResult:
    """RRF 병합 시 사용하는 단일 결과 + 누적 RRF 점수."""

    doc_id: str
    score: float
    payload: Any


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------


async def generate_hyde_doc(
    query: str,
    *,
    http_client: Any,
    ollama_model: str,
    api_base: str,
    request_timeout: float = 30.0,
    max_tokens: int = 400,
    keep_alive: str = "5m",
) -> str:
    """질의에 대한 가상 문서를 LLM으로 생성합니다 — never raises.

    실패 시 빈 문자열 반환 (호출 측이 원본 질의로 fallback).
    """
    if not query or not query.strip():
        return ""

    prompt = _HYDE_PROMPT.format(query=query.strip())
    try:
        response = await http_client.post(
            f"{api_base}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": keep_alive,
                "options": {"num_predict": max_tokens},
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "") or data.get("thinking", "")
    except Exception as exc:
        logger.warning("HyDE 생성 실패: %s — 원본 질의로 fallback", exc)
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return cleaned[: max_tokens * 4]  # 대략적 안전 상한


# ---------------------------------------------------------------------------
# Multi-Query Expansion
# ---------------------------------------------------------------------------


async def generate_multi_queries(
    query: str,
    *,
    http_client: Any,
    ollama_model: str,
    api_base: str,
    n: int = 3,
    request_timeout: float = 30.0,
    max_tokens: int = 300,
    keep_alive: str = "5m",
) -> list[str]:
    """원본 질의를 N개 패러프레이즈로 변형 — never raises.

    실패 시 빈 리스트 반환. 결과에는 원본은 포함되지 않으므로 호출 측이
    원본 + 변형들을 함께 검색에 사용.
    """
    if not query or not query.strip() or n <= 0:
        return []

    prompt = _MULTI_QUERY_PROMPT.format(query=query.strip(), n=n)
    try:
        response = await http_client.post(
            f"{api_base}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": "json",
                "keep_alive": keep_alive,
                "options": {"num_predict": max_tokens},
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "") or data.get("thinking", "")
    except Exception as exc:
        logger.warning("Multi-query 생성 실패: %s — 원본만 사용", exc)
        return []

    parsed = _parse_multi_query_json(raw)
    if not parsed:
        return []
    seen: set[str] = set()
    out: list[str] = []
    normalized_query = " ".join(query.lower().split())
    for q in parsed:
        if not isinstance(q, str):
            continue
        cleaned = q.strip()
        if not cleaned:
            continue
        norm = " ".join(cleaned.lower().split())
        if norm == normalized_query or norm in seen:
            continue
        seen.add(norm)
        out.append(cleaned)
        if len(out) >= n:
            break
    return out


def _parse_multi_query_json(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return []
    try:
        obj = json.loads(cleaned[brace_start : brace_end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    queries = obj.get("queries", [])
    if isinstance(queries, list):
        return [q for q in queries if isinstance(q, str)]
    return []


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_merge(
    result_lists: list[list[Any]],
    *,
    top_k: int,
    k: int = 60,
    id_attr: str = "doc_id",
) -> list[Any]:
    """여러 검색 결과 리스트를 Reciprocal Rank Fusion으로 병합합니다.

    표준 RRF 공식: score(d) = sum_i 1 / (k + rank_i(d))
    rank_i는 1부터 시작.

    Parameters
    ----------
    result_lists:
        각각 정렬된 결과 객체 리스트. 객체는 ``id_attr``로 ID 추출.
    top_k:
        반환할 최대 결과 수.
    k:
        RRF 상수 (논문 기본값 60).
    id_attr:
        ID로 사용할 객체의 속성 이름.

    Returns
    -------
    list
        RRF 점수 내림차순으로 정렬된 객체 목록 (각 ID당 1개 객체 — 가장 처음 등장한 인스턴스).
    """
    if not result_lists:
        return []
    if k <= 0:
        k = 60
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    seen: dict[str, Any] = {}
    for rlist in result_lists:
        for rank, item in enumerate(rlist, start=1):
            doc_id = str(getattr(item, id_attr, "") or "")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in seen:
                seen[doc_id] = item

    ranked_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [seen[d] for d in ranked_ids[:top_k]]


__all__ = [
    "generate_hyde_doc",
    "generate_multi_queries",
    "rrf_merge",
]
