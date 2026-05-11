"""Skill dataclass 및 trigger 매칭 로직."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    """도메인 지식 팩.

    Attributes
    ----------
    name:
        skill 고유 식별자.
    description:
        사람이 읽을 수 있는 설명.
    triggers:
        질의에 포함되어야 skill이 활성화되는 키워드·정규식 목록.
        - plain string: 대소문자 무시 substring 매칭
        - ``regex:...``: 정규식 매칭
    prompt_addon:
        synthesis prompt에 주입할 텍스트 블록. 사용자 의도·규약을 담음.
    priority:
        복수 skill이 동시 매칭될 때의 우선순위. 높은 값이 먼저 적용.
    """

    name: str
    description: str = ""
    triggers: tuple[str, ...] = field(default_factory=tuple)
    prompt_addon: str = ""
    priority: int = 0

    def matches(self, query: str) -> bool:
        """질의에 skill의 trigger가 매칭되는지 검사."""
        if not self.triggers:
            return False
        lowered = query.lower()
        for t in self.triggers:
            if t.startswith("regex:"):
                pattern = t[len("regex:"):]
                try:
                    if re.search(pattern, query, re.IGNORECASE):
                        return True
                except re.error:
                    continue
            else:
                if t.lower() in lowered:
                    return True
        return False


__all__ = ["Skill"]
