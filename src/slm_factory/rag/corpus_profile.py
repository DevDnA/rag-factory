"""CorpusProfile — RAG 인덱스가 어떤 도메인을 담고 있는지의 자기 기술.

slm-factory는 어떤 도메인의 문서가 들어올지 사전에 알 수 없습니다(의료·법률·
RFP·금융 등). 따라서 라우팅·합성 단계에서 "이 corpus는 무엇을 다루는가"라는
컨텍스트가 있어야 도메인 약어·전문어를 정확히 분류할 수 있습니다.

본 모듈은:
1. 인덱스 시 첫 N개 청크를 표본으로 LLM에 요약을 요청해 ``CorpusProfile`` 생성
2. ``corpus_profile.json``에 영속화
3. 서버 시작 시 로드해 ``IntentClassifier`` 등 라우팅 컴포넌트에 주입

생성 실패는 absorb — 빈 profile로 동작(현재 라우팅과 동일).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger("rag.corpus_profile")


@dataclass(frozen=True)
class CorpusProfile:
    """인덱스 corpus의 도메인 자기 기술.

    Attributes
    ----------
    name:
        한 줄 명칭 — 예: "한국 통신사 RFP 문서". 빈 문자열이면 미설정.
    summary:
        2~5문장 요약. 라우팅·합성 프롬프트 헤더에 주입됩니다.
    keywords:
        도메인 핵심 키워드/약어 — 예: ["NMS", "BIS", "4×4 MIMO"].
        IntentClassifier가 약어 false negative를 줄이는 데 사용.
    generated_at:
        ISO-8601 생성 시각.
    model:
        프로파일을 생성한 LLM 모델명.
    chunks_sampled:
        프로파일 생성에 사용된 표본 청크 수.
    """

    name: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    generated_at: str = ""
    model: str = ""
    chunks_sampled: int = 0

    def is_empty(self) -> bool:
        """name·summary·keywords가 모두 비어 있으면 True (라우팅 컨텍스트 미주입)."""
        return not self.name and not self.summary and not self.keywords

    def to_prompt_header(self) -> str:
        """IntentClassifier·합성 프롬프트에 주입할 헤더 텍스트.

        빈 profile이면 빈 문자열을 반환합니다 — 호출 측이 그대로 concat하면 됩니다.
        """
        if self.is_empty():
            return ""
        lines: list[str] = ["[본 corpus 도메인 정보]"]
        if self.name:
            lines.append(f"- 명칭: {self.name}")
        if self.summary:
            lines.append(f"- 요약: {self.summary}")
        if self.keywords:
            lines.append(f"- 핵심 키워드: {', '.join(self.keywords[:20])}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 영속화
# ---------------------------------------------------------------------------


def load_corpus_profile(path: Path) -> CorpusProfile:
    """JSON 파일에서 CorpusProfile을 로드합니다 — never raises.

    파일이 없거나 파싱 실패 시 빈 profile을 반환합니다.
    """
    if not path.exists():
        return CorpusProfile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("CorpusProfile 로드 실패 (%s): %s — 빈 profile 사용", path, exc)
        return CorpusProfile()

    keywords = data.get("keywords", []) or []
    if not isinstance(keywords, list):
        keywords = []

    return CorpusProfile(
        name=str(data.get("name", "") or ""),
        summary=str(data.get("summary", "") or ""),
        keywords=[str(k) for k in keywords if k],
        generated_at=str(data.get("generated_at", "") or ""),
        model=str(data.get("model", "") or ""),
        chunks_sampled=int(data.get("chunks_sampled", 0) or 0),
    )


def save_corpus_profile(profile: CorpusProfile, path: Path) -> None:
    """CorpusProfile을 JSON 파일로 영속화합니다 — never raises.

    저장 실패는 로그만 남기고 호출 측에 전파하지 않습니다(인덱싱·서버 가용성 우선).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("CorpusProfile 저장 실패 (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# 자동 생성기
# ---------------------------------------------------------------------------


_PROFILE_PROMPT = """다음은 어떤 RAG 인덱스에서 추출한 표본 청크 {n}개입니다.
이 corpus가 어떤 도메인을 다루는지 분석하여 JSON으로 답하세요.

[표본 청크]
{samples}

규칙:
- "name": 도메인을 한 줄로 명명 (예: "한국 통신사 RFP 문서", "산부인과 진료 가이드라인").
- "summary": 2~4문장으로 corpus가 다루는 주제·범위를 요약.
- "keywords": 도메인을 식별하는 핵심 키워드·약어 8~15개. **약어와 고유명사를 우선** 포함.
- 모든 필드는 한국어로 작성.
- 청크에 없는 정보는 추측하지 마세요.

반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 금지):
{{
  "name": "...",
  "summary": "...",
  "keywords": ["...", "...", "..."]
}}
"""


def _format_samples(chunks: list[str], max_chars_per_chunk: int = 400) -> str:
    """청크 목록을 LLM 프롬프트에 넣을 텍스트로 포맷합니다."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = (chunk or "").strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        parts.append(f"[청크 {i}]\n{text}")
    return "\n\n".join(parts)


def _parse_profile_json(raw: str) -> dict | None:
    """LLM 응답에서 JSON 객체를 추출합니다 — 실패 시 None."""
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return None
    try:
        parsed = json.loads(cleaned[brace_start : brace_end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


async def generate_corpus_profile(
    *,
    chunks: list[str],
    http_client: Any,
    ollama_model: str,
    api_base: str,
    request_timeout: float = 60.0,
    keep_alive: str = "5m",
    max_tokens: int = 800,
) -> CorpusProfile:
    """LLM으로 CorpusProfile을 자동 생성합니다 — never raises.

    Parameters
    ----------
    chunks:
        표본 청크 텍스트 목록. 권장 8~16개.
    http_client:
        Ollama ``/api/generate``를 호출할 ``httpx.AsyncClient``.
    ollama_model:
        프로파일 생성에 사용할 모델명.
    api_base:
        Ollama API 베이스 URL.

    Returns
    -------
    CorpusProfile
        생성된 profile. 실패 시 빈 profile.
    """
    from datetime import datetime, timezone

    if not chunks:
        return CorpusProfile()

    prompt = _PROFILE_PROMPT.format(
        n=len(chunks),
        samples=_format_samples(chunks),
    )

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
        logger.warning("CorpusProfile LLM 호출 실패: %s — 빈 profile 사용", exc)
        return CorpusProfile()

    parsed = _parse_profile_json(raw)
    if parsed is None:
        logger.warning("CorpusProfile JSON 파싱 실패 — 빈 profile 사용")
        return CorpusProfile()

    raw_keywords = parsed.get("keywords", []) or []
    if not isinstance(raw_keywords, list):
        raw_keywords = []
    keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]

    return CorpusProfile(
        name=str(parsed.get("name", "") or "").strip(),
        summary=str(parsed.get("summary", "") or "").strip(),
        keywords=keywords,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=ollama_model,
        chunks_sampled=len(chunks),
    )


def merge_with_override(
    auto: CorpusProfile,
    *,
    name_override: str = "",
    summary_override: str = "",
    keywords_override: list[str] | None = None,
) -> CorpusProfile:
    """자동 생성 profile에 사용자 override를 적용합니다.

    각 필드별로 override가 비어 있지 않으면 우선시합니다. keywords는 override가
    제공되면 완전 대체(append 아님) — 사용자가 의도한 키워드만 노출되도록.
    """
    keywords = (
        list(keywords_override)
        if keywords_override is not None and len(keywords_override) > 0
        else list(auto.keywords)
    )
    return CorpusProfile(
        name=name_override.strip() or auto.name,
        summary=summary_override.strip() or auto.summary,
        keywords=keywords,
        generated_at=auto.generated_at,
        model=auto.model,
        chunks_sampled=auto.chunks_sampled,
    )


__all__ = [
    "CorpusProfile",
    "generate_corpus_profile",
    "load_corpus_profile",
    "merge_with_override",
    "save_corpus_profile",
]
