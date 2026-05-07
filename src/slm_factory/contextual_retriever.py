"""Anthropic Contextual Retrieval — 청크에 LLM 생성 컨텍스트를 prefix로 부여합니다.

각 청크에 대해 Teacher LLM에 "이 청크가 전체 문서에서 어떤 위치/맥락으로 등장하는지"를
50~100토큰으로 요약하게 한 뒤, 그 결과를 청크 본문 앞에 덧붙입니다.
임베딩·BM25·리랭커 모두 동일한 prefix를 보게 되어 검색 정확도가 향상됩니다.

참고: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .utils import get_logger

if TYPE_CHECKING:
    from .config import ContextualRetrievalConfig
    from .teacher.base import BaseTeacher

logger = get_logger("contextual_retriever")


_KO_PROMPT = """다음은 전체 문서입니다.
<document>
{document}
</document>

다음은 위 문서에서 추출한 일부 청크입니다.
<chunk>
{chunk}
</chunk>

이 청크가 전체 문서 안에서 어떤 위치·맥락으로 등장하는지 검색 정확도를 높이기 위해
한국어로 1~3문장(최대 {max_chars}자) 이내로 간결하게 요약해 주세요.
청크의 내용 자체를 다시 쓰지 말고, 어떤 섹션·주제·전제 아래에서 다뤄지는지만 적으세요.
요약문 외의 어떤 텍스트도 출력하지 마세요."""


_EN_PROMPT = """Here is the full document.
<document>
{document}
</document>

Here is a chunk extracted from the document.
<chunk>
{chunk}
</chunk>

Please write a short, succinct context (1-3 sentences, at most {max_chars} chars) that
situates this chunk within the overall document — what section/topic/assumption it
belongs to — to improve retrieval. Do not restate the chunk content itself.
Answer only with the context, nothing else."""


@dataclass
class ChunkContext:
    """청크와 LLM이 생성한 컨텍스트의 쌍입니다."""

    chunk_index: int
    """청크의 corpus_rows 인덱스."""

    context: str
    """생성된 컨텍스트 prefix (실패 시 빈 문자열)."""


def _build_prompt(document: str, chunk: str, max_chars: int, language: str) -> str:
    template = _KO_PROMPT if language == "ko" else _EN_PROMPT
    return template.format(document=document, chunk=chunk, max_chars=max_chars)


def _truncate_document(document: str, max_chars: int) -> str:
    """부모 문서를 LLM에 전달하기 전에 절단합니다. 토큰이 아닌 문자 기준입니다."""
    if len(document) <= max_chars:
        return document
    half = max_chars // 2
    return f"{document[:half]}\n... [중략] ...\n{document[-half:]}"


async def generate_chunk_contexts_async(
    teacher: BaseTeacher,
    document: str,
    chunks: list[str],
    config: ContextualRetrievalConfig,
    language: str = "ko",
) -> list[str]:
    """문서 내 모든 청크에 대한 컨텍스트를 병렬로 생성합니다.

    매개변수
    ----------
    teacher:
        ``BaseTeacher`` 인스턴스 (``agenerate`` 사용).
    document:
        부모 문서 전체 텍스트. ``config.doc_truncate_chars``로 절단됩니다.
    chunks:
        컨텍스트를 생성할 청크 본문 리스트.
    config:
        ``ContextualRetrievalConfig``.
    language:
        프롬프트 언어 (``"ko"`` 또는 ``"en"``).

    반환값
    -------
    list[str]
        각 청크에 대응하는 컨텍스트 prefix 리스트. 실패한 항목은 빈 문자열입니다.
    """
    if not chunks:
        return []

    if len(document) < config.skip_short_docs:
        logger.debug(
            "문서가 짧아(%d < %d) 컨텍스트 생성을 생략합니다",
            len(document),
            config.skip_short_docs,
        )
        return ["" for _ in chunks]

    truncated_doc = _truncate_document(document, config.doc_truncate_chars)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _one(chunk: str) -> str:
        async with semaphore:
            prompt = _build_prompt(
                truncated_doc, chunk, config.max_chars, language
            )
            try:
                response = await teacher.agenerate(
                    prompt,
                    temperature=0.0,
                    think=False,
                )
            except Exception as exc:
                logger.warning(
                    "청크 컨텍스트 생성 실패 (%s) — 빈 prefix 사용",
                    type(exc).__name__,
                )
                return ""
            cleaned = (response or "").strip()
            if not cleaned:
                return ""
            if len(cleaned) > config.max_chars:
                cleaned = cleaned[: config.max_chars].rstrip() + "…"
            return cleaned

    tasks = [_one(c) for c in chunks]
    results = await asyncio.gather(*tasks)
    return list(results)


def prepend_context(content: str, context: str) -> str:
    """청크 본문 앞에 컨텍스트를 ``[Context] ... \\n`` 형식으로 부여합니다.

    빈 컨텍스트면 본문을 그대로 반환합니다.
    """
    if not context:
        return content
    return f"[Context] {context}\n\n{content}"
