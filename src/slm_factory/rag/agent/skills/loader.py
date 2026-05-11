"""디렉터리에서 skill YAML 파일을 로드하고 쿼리에 매칭하는 registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ....utils import get_logger
from .base import Skill

logger = get_logger("rag.agent.skills.loader")


class SkillRegistry:
    """로드된 skill 컬렉션 + 질의 매칭.

    Parameters
    ----------
    skills:
        등록할 ``Skill`` 인스턴스 목록. ``load_skills_from_dir()``가 반환한 것을 넣음.
    """

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: list[Skill] = list(skills)

    def all(self) -> list[Skill]:
        """등록된 모든 skill 목록."""
        return list(self._skills)

    def select_for_query(self, query: str, limit: int | None = None) -> list[Skill]:
        """질의에 매칭되는 skill들을 우선순위 내림차순으로 반환.

        Parameters
        ----------
        query:
            사용자 질의.
        limit:
            최대 반환 개수. ``None``이면 제한 없음.
        """
        matches = [s for s in self._skills if s.matches(query)]
        matches.sort(key=lambda s: s.priority, reverse=True)
        if limit is not None:
            return matches[:limit]
        return matches

    @staticmethod
    def format_addons(skills: list[Skill]) -> str:
        """synthesis prompt에 주입할 addon 블록을 생성합니다."""
        if not skills:
            return ""
        lines: list[str] = ["[도메인 스킬]"]
        for s in skills:
            if not s.prompt_addon.strip():
                continue
            lines.append(f"## {s.name}")
            lines.append(s.prompt_addon.strip())
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._skills)


def load_skills_from_dir(skills_dir: str | os.PathLike[str]) -> list[Skill]:
    """디렉터리의 YAML 파일들을 파싱하여 Skill 목록을 반환.

    디렉터리가 없거나 비어 있으면 빈 목록. YAML 파싱 오류는 개별 skill을 건너뛰고
    나머지를 로드합니다 (never-raise 유지).
    """
    path = Path(skills_dir)
    if not path.is_dir():
        logger.debug("skills_dir 없음: %s — 빈 목록 반환", path)
        return []

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml이 설치되지 않았습니다 — skills 로드를 건너뜁니다")
        return []

    skills: list[Skill] = []
    for fp in sorted(path.rglob("*.yaml")) + sorted(path.rglob("*.yml")):
        try:
            data = yaml.safe_load(fp.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("skill YAML 파싱 실패 (%s): %s — 건너뜀", fp, exc)
            continue
        skill = _build_skill(data, source=str(fp))
        if skill is not None:
            skills.append(skill)
    return skills


def _build_skill(data, *, source: str) -> Skill | None:
    """dict → Skill. 잘못된 형식은 None 반환."""
    if not isinstance(data, dict):
        logger.debug("skill 파일 %s: dict가 아님 — 건너뜀", source)
        return None
    name = str(data.get("name", "")).strip()
    if not name:
        logger.debug("skill 파일 %s: name 없음 — 건너뜀", source)
        return None

    raw_triggers = data.get("triggers") or []
    if not isinstance(raw_triggers, (list, tuple)):
        raw_triggers = []
    triggers = tuple(str(t).strip() for t in raw_triggers if str(t).strip())

    description = str(data.get("description", ""))[:500]
    prompt_addon = str(data.get("prompt_addon", ""))
    try:
        priority = int(data.get("priority", 0))
    except (TypeError, ValueError):
        priority = 0

    return Skill(
        name=name,
        description=description,
        triggers=triggers,
        prompt_addon=prompt_addon,
        priority=priority,
    )


__all__ = ["SkillRegistry", "load_skills_from_dir"]
