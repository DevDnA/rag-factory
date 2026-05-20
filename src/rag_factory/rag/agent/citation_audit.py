"""인용 토큰 audit (Phase 14).

합성된 답변에서 ``[doc:파일명]`` 형식의 인용 토큰을 추출하고, 실제 source list의
``doc_id``와 대조해 *환각 인용*을 찾아냅니다. orchestrator가 이 결과를 SSE
``warning`` 이벤트로 발행합니다.

설계 결정
---------
- **답변 본문을 변형하지 않음**: 환각 인용이 있어도 답변은 그대로 사용자에게
  전달. 사용자 경험을 위해 warning은 reasoning panel에만 표시.
- **prefix 매칭**: source의 ``doc_id``가 ``"rfp.pdf::p5"`` 같이 chunk suffix를
  포함할 수 있으므로 인용 토큰 ``[doc:rfp.pdf]``는 doc_id의 ``"::""`` 앞부분과
  매칭합니다.
- **case-insensitive**: LLM이 파일명을 대소문자 변형해 인용할 수 있으므로
  정규화 비교.
"""

from __future__ import annotations

import re

# `[doc:파일명]` — 파일명 부분은 닫는 대괄호와 콜론을 제외한 모든 문자.
_CITATION_TOKEN = re.compile(r"\[doc:([^\]]+)\]")


def extract_citations(answer: str) -> list[str]:
    """답변 본문에서 인용된 doc 파일명 목록을 추출합니다 (중복 제거).

    Examples
    --------
    >>> extract_citations("동의 요건 [doc:law.pdf]. 세부는 [doc:guide.pdf].")
    ['law.pdf', 'guide.pdf']
    >>> extract_citations("출처 없음")
    []
    """
    if not answer:
        return []
    raw = [match.group(1).strip() for match in _CITATION_TOKEN.finditer(answer)]
    seen: set[str] = set()
    unique: list[str] = []
    for token in raw:
        if not token or token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def audit_citations(answer: str, sources: list[dict]) -> list[str]:
    """답변의 인용 토큰 중 source list에 매칭되지 않는 doc 이름을 반환합니다.

    Parameters
    ----------
    answer:
        합성된 답변 본문.
    sources:
        ``[{"content": ..., "doc_id": ..., "score": ...}, ...]`` 형태의 source list.

    Returns
    -------
    list[str]
        source list와 매칭되지 않은 환각 인용 doc 이름 (중복 제거, 답변 순서).

    Notes
    -----
    - source ``doc_id``는 chunk suffix를 포함할 수 있음 (``"rfp.pdf::p5"``,
      ``"rfp.pdf#chunk_3"``). 매칭은 ``::``/``#`` 앞부분 (case-insensitive)으로
      수행합니다.
    - source list가 비어 있으면 답변의 모든 인용이 환각으로 간주됩니다.
    """
    cited = extract_citations(answer)
    if not cited:
        return []

    def _canon(s: str) -> str:
        """매칭용 정규화 — chunk suffix 제거, 공백 단순화, 소문자.

        한글 파일명에서 LLM이 ``"사업(최종)"`` → ``"사업 (최종)"`` 처럼 공백을
        삽입/제거하는 경우를 흡수합니다.
        """
        if not s:
            return ""
        base = re.split(r"::|#", s, maxsplit=1)[0]
        return re.sub(r"\s+", "", base).strip().lower()

    known: set[str] = set()
    for src in sources:
        # Phase 14 — source_doc_id (원본 파일명)이 인용 매칭의 우선순위.
        # UUID ``doc_id``만 있던 기존 데이터와도 호환 유지 (둘 다 known에 등록).
        candidates = [
            src.get("source_doc_id"),
            src.get("doc_id"),
            src.get("source"),
        ]
        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                continue
            stripped = _canon(cand)
            if stripped:
                known.add(stripped)

    missing: list[str] = []
    for token in cited:
        normalized = _canon(token)
        if not normalized:
            continue
        # 정확 매칭 우선, 그 다음 prefix 매칭 (양방향). LLM이 긴 파일명을
        # 일부만 인용하거나 (예: "3. 제안요청서") 확장자를 누락한 경우 처리.
        # 짧은 prefix(< 4자)는 우연 매칭 위험이 있어 정확 매칭만 허용.
        matched = normalized in known
        if not matched and len(normalized) >= 4:
            matched = any(
                (len(k) >= 4 and (normalized.startswith(k) or k.startswith(normalized)))
                for k in known
            )
        if not matched:
            missing.append(token)
    return missing


__all__ = ["extract_citations", "audit_citations"]
