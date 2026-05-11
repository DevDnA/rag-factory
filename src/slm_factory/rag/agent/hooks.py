"""파이프라인 lifecycle Hooks — 주요 지점의 pre/post 처리.

oh-my-openagent의 50+ hook 시스템을 RAG 파이프라인의 핵심 지점에만
적용한 경량 버전. 사용자가 질의 정규화, source 필터링, 답변 후처리 등을
코어 코드 변경 없이 삽입할 수 있습니다.

Hook 지점
--------
- ``pre_query``: 질의 정규화·확장 — ``(query: str) -> str``
- ``post_search``: source 필터링·정렬 — ``(sources: list[dict]) -> list[dict]``
- ``post_synthesis``: 답변 후처리 — ``(answer: str) -> str``

설계 원칙
---------
- **Never-raise**: 각 hook은 예외를 호출 측에 전파하지 않음. Registry가 예외를
  삼키고 이전 단계 결과를 유지.
- **Idempotent 권장**: 동일 입력에 동일 출력을 반환해야 재실행 시 안정.
- **Sync or async 모두 허용**: Registry가 자동으로 대응.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable

from ...utils import get_logger

logger = get_logger("rag.agent.hooks")


Hook = Callable[[Any], Any]  # sync 또는 async
"""hook 시그니처. 입력 payload를 받아 새 payload 반환."""


class HookRegistry:
    """lifecycle hook 등록·실행 관리자.

    Hook 시그니처
    -------------
    fn 시그니처: ``(payload: Any) -> Any | Awaitable[Any]``.
    fn 자체는 ``callable``이어야 하며 (coroutine factory 함수 또는 일반 함수),
    호출 결과가 coroutine/awaitable이면 자동으로 ``await``됩니다.
    이미 만들어진 coroutine 객체나 ``asyncio.Task``를 그대로 등록하면 안 됩니다.
    fn은 새 payload를 반환해야 하며, ``None``을 반환하면 그대로 다음 hook에
    전달됩니다 (``None``도 유효한 payload).

    예외는 Registry가 삼키고 직전 결과를 유지하므로 hook은 raise해도 안전합니다.

    Parameters
    ----------
    enabled:
        ``False``이면 ``run()``이 payload를 그대로 반환 (no-op).
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._hooks: dict[str, list[Hook]] = {}

    def register(self, point: str, fn: Hook) -> None:
        """hook 지점에 함수를 등록. 등록 순서대로 실행됩니다."""
        self._hooks.setdefault(point, []).append(fn)

    def clear(self, point: str | None = None) -> None:
        """특정 지점 또는 전체 hook 제거."""
        if point is None:
            self._hooks.clear()
        else:
            self._hooks.pop(point, None)

    async def run(self, point: str, payload: Any) -> Any:
        """모든 hook을 순서대로 실행하고 최종 결과 반환 — never raises."""
        if not self._enabled:
            return payload
        result = payload
        for fn in self._hooks.get(point, []):
            try:
                out = fn(result)
                if inspect.isawaitable(out):
                    out = await out
                result = out
            except Exception as exc:
                logger.warning(
                    "Hook %s.%s 실패: %s — 건너뜀",
                    point,
                    getattr(fn, "__name__", repr(fn)),
                    exc,
                )
                # 이전 result 유지하고 다음 hook으로
        return result

    def count(self, point: str) -> int:
        """등록된 hook 개수."""
        return len(self._hooks.get(point, []))


# ---------------------------------------------------------------------------
# 내장 hooks — 곧바로 등록해서 사용 가능한 helper들
# ---------------------------------------------------------------------------


def normalize_korean_whitespace(query: str) -> str:
    """공백·개행 정규화. 연속 공백을 하나로, 양끝 공백 제거."""
    if not isinstance(query, str):
        return query
    return re.sub(r"\s+", " ", query).strip()


def dedup_sources_by_doc_id(sources: list[dict]) -> list[dict]:
    """source 목록에서 doc_id 기준 중복 제거. 순서 보존, 첫 등장만 유지."""
    if not isinstance(sources, list):
        return sources
    seen: set[str] = set()
    out: list[dict] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        key = str(s.get("doc_id", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def strip_html_from_answer(answer: str) -> str:
    """답변에서 HTML 태그 잔재 제거 (prompt가 이미 금지하지만 이중 방어)."""
    if not isinstance(answer, str):
        return answer
    return re.sub(r"<[^>]+>", "", answer)


BUILT_IN_HOOKS: dict[str, Hook] = {
    "normalize_korean_whitespace": normalize_korean_whitespace,
    "dedup_sources_by_doc_id": dedup_sources_by_doc_id,
    "strip_html_from_answer": strip_html_from_answer,
}
"""이름으로 참조 가능한 내장 hook 사전."""


def build_default_registry(
    *, enabled: bool = True, builtin_names: list[str] | None = None
) -> HookRegistry:
    """config.builtin_hooks로 지정된 내장 hook을 기본 지점에 등록한 registry 반환."""
    reg = HookRegistry(enabled=enabled)
    if not enabled or not builtin_names:
        return reg
    for name in builtin_names:
        fn = BUILT_IN_HOOKS.get(name)
        if fn is None:
            logger.warning("알 수 없는 내장 hook: %s — 건너뜁니다", name)
            continue
        point = _default_point_for(name)
        if point is None:
            continue
        reg.register(point, fn)
    return reg


def _default_point_for(hook_name: str) -> str | None:
    """내장 hook의 기본 지점 매핑."""
    if hook_name == "normalize_korean_whitespace":
        return "pre_query"
    if hook_name == "dedup_sources_by_doc_id":
        return "post_search"
    if hook_name == "strip_html_from_answer":
        return "post_synthesis"
    return None


__all__ = [
    "Hook",
    "HookRegistry",
    "BUILT_IN_HOOKS",
    "build_default_registry",
    "normalize_korean_whitespace",
    "dedup_sources_by_doc_id",
    "strip_html_from_answer",
]
