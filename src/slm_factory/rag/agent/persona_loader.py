"""Phase 14 — YAML로 사용자 정의 persona 로딩.

사용자는 디렉터리의 YAML 파일로 자신의 도메인에 맞는 persona를 정의하고,
intent별로 built-in persona 대신 활성화할 수 있습니다.

YAML 형식
--------
```yaml
name: legal-expert
description: 법률 전문가 페르소나
intent: factual                   # 이 intent에 매칭되면 해당 persona 사용
allowed_tools: [search, lookup]   # 도구 화이트리스트
plan_strategy_hint: fact
synthesis_prompt_template: |
  당신은 법률 전문가입니다.
  ... (history/context/query placeholder 포함)
```
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ...utils import get_logger
from .personas.base import Persona

logger = get_logger("rag.agent.persona_loader")


class CustomPersona(Persona):
    """동적으로 생성된 persona — YAML 데이터를 담는 wrapper.

    다른 persona와 달리 인스턴스 속성으로 정의(클래스 속성 아님).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        allowed_tools: frozenset[str] | None = None,
        synthesis_prompt_template: str | None = None,
        plan_strategy_hint: str | None = None,
        intent: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.allowed_tools = allowed_tools
        self.synthesis_prompt_template = synthesis_prompt_template
        self.plan_strategy_hint = plan_strategy_hint
        self.intent = intent


def load_custom_personas(
    persona_dir: str | os.PathLike[str],
) -> list[CustomPersona]:
    """디렉터리의 YAML 파일을 파싱하여 CustomPersona 목록을 반환합니다 — never raises."""
    path = Path(persona_dir)
    if not path.is_dir():
        return []

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml 미설치 — custom personas 로드 건너뜀")
        return []

    personas: list[CustomPersona] = []
    for fp in sorted(path.rglob("*.yaml")) + sorted(path.rglob("*.yml")):
        try:
            data = yaml.safe_load(fp.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("custom persona 파싱 실패 (%s): %s", fp, exc)
            continue
        p = _build_persona(data, source=str(fp))
        if p is not None:
            personas.append(p)
    return personas


def _build_persona(data, *, source: str) -> CustomPersona | None:
    """dict → CustomPersona. 필수 필드(name, synthesis_prompt_template) 검증."""
    if not isinstance(data, dict):
        logger.debug("custom persona %s: dict 아님", source)
        return None
    name = str(data.get("name", "")).strip()
    if not name:
        logger.debug("custom persona %s: name 없음", source)
        return None

    template = data.get("synthesis_prompt_template")
    if template is not None and not isinstance(template, str):
        logger.warning("custom persona %s: synthesis_prompt_template은 문자열이어야 함", source)
        template = None
    if template and ("{query}" not in template or "{context}" not in template):
        logger.warning(
            "custom persona '%s': template에 {query} 또는 {context} placeholder 없음 — 적용 안 될 수 있음",
            name,
        )

    raw_tools = data.get("allowed_tools")
    if raw_tools is None:
        allowed: frozenset[str] | None = None
    elif isinstance(raw_tools, (list, tuple)):
        allowed = frozenset(str(t).strip() for t in raw_tools if str(t).strip())
    else:
        allowed = None

    intent = data.get("intent")
    if intent is not None:
        intent = str(intent).strip() or None

    hint = data.get("plan_strategy_hint")
    if hint is not None:
        hint = str(hint).strip() or None

    return CustomPersona(
        name=name,
        description=str(data.get("description", ""))[:500],
        allowed_tools=allowed,
        synthesis_prompt_template=template,
        plan_strategy_hint=hint,
        intent=intent,
    )


class CustomPersonaRegistry:
    """Custom personas를 intent별로 조회 가능한 registry."""

    def __init__(self, personas: Iterable[CustomPersona] = ()) -> None:
        self._by_intent: dict[str, CustomPersona] = {}
        self._all: list[CustomPersona] = list(personas)
        for p in self._all:
            if p.intent:
                self._by_intent[p.intent] = p

    def select_for_intent(self, intent: str | None) -> CustomPersona | None:
        if intent is None:
            return None
        return self._by_intent.get(str(intent))

    def all(self) -> list[CustomPersona]:
        return list(self._all)

    def __len__(self) -> int:
        return len(self._all)


__all__ = ["CustomPersona", "CustomPersonaRegistry", "load_custom_personas"]
