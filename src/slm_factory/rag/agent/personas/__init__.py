"""Agent persona 시스템 — 의도별 전용 에이전트 구성.

oh-my-openagent의 전문 agent 패턴을 RAG Q&A에 적용. 각 persona는:

- 고유 system prompt
- 사용 가능한 도구 화이트리스트
- 답변 synthesis 스타일
- 자체 검증 기준 (예: Clarifier는 sources 없이 questions 반환)

Persona는 ``IntentCategory``와 1:1 또는 N:1 매핑되며, orchestrator가
``IntentClassifier``의 결과에 따라 적절한 persona를 선택합니다.
"""

from __future__ import annotations

from .analyst import Analyst
from .base import Persona, PersonaResult
from .clarifier import Clarifier
from .comparator import Comparator
from .procedural import Procedural
from .researcher import Researcher

__all__ = [
    "Persona",
    "PersonaResult",
    "Analyst",
    "Clarifier",
    "Comparator",
    "Procedural",
    "Researcher",
]
