"""Intent → Persona 매퍼.

``IntentClassifier``의 결과를 받아 적절한 agent persona를 선택합니다.
``intent`` 매핑이 없거나 ``personas_enabled=False``인 경우 ``None``을 반환하며,
orchestrator는 기본 synthesis 프롬프트로 fallback합니다.

Clarifier는 이 라우터를 거치지 않고 orchestrator의 ``handle_auto``에서
직접 ambiguous 의도를 처리합니다 (응답 형식이 다르기 때문).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .personas.analyst import Analyst
from .personas.base import Persona
from .personas.comparator import Comparator
from .personas.procedural import Procedural
from .personas.researcher import Researcher

if TYPE_CHECKING:
    from .intent_classifier import IntentCategory


# Intent → Persona 클래스 매핑. Clarifier는 의도적으로 제외 (별도 경로).
_INTENT_TO_PERSONA: dict[str, type[Persona]] = {
    "factual": Researcher,
    "comparative": Comparator,
    "analytical": Analyst,
    "procedural": Procedural,
    "exploratory": Analyst,  # 탐색 질의는 종합형 Analyst가 적합
}


class PersonaRouter:
    """의도 기반 persona 선택기.

    Parameters
    ----------
    enabled:
        ``False``이면 ``select()``가 항상 ``None`` 반환 → 기본 synthesis 사용.
    custom_registry:
        Phase 14 — YAML로 로드한 사용자 정의 persona registry. 매칭되는
        custom persona가 있으면 built-in보다 우선 반환.
    """

    def __init__(
        self,
        enabled: bool = False,
        custom_registry=None,
    ) -> None:
        self._enabled = enabled
        self._custom_registry = custom_registry

    def select(self, intent: "IntentCategory | str | None") -> Persona | None:
        """의도에 맞는 persona 인스턴스를 반환합니다.

        Custom persona 매핑이 있으면 우선 반환, 없으면 built-in 매핑을 사용.
        """
        if not self._enabled or intent is None:
            return None
        # Phase 14: custom persona 우선
        if self._custom_registry is not None:
            custom = self._custom_registry.select_for_intent(str(intent))
            if custom is not None:
                return custom

        cls = _INTENT_TO_PERSONA.get(str(intent))
        if cls is None:
            return None
        return cls()

    @staticmethod
    def available_personas() -> list[str]:
        """등록된 persona 이름 목록 — 디버깅용."""
        return sorted({cls().name for cls in _INTENT_TO_PERSONA.values()})


__all__ = ["PersonaRouter"]
