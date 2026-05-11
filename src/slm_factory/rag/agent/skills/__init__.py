"""Skills 시스템 — 도메인 지식 팩을 YAML로 정의하고 자동 주입.

oh-my-openagent의 skills (git-master, frontend-ui-ux) 패턴을 RAG Q&A
컨텍스트에 적용. 사용자는 YAML 파일로 도메인 지식을 패킹하고,
``triggers``에 매칭되는 질의에 자동으로 prompt addon이 주입됩니다.

Skill 파일 구조
---------------
```yaml
name: legal
description: 법률 도메인 인용·형식
triggers: [조항, 법, 약관, 판례]
prompt_addon: |
  법률 답변 시:
  - 조항은 "제X조제Y항" 형식으로
  - 판례는 "[사건번호] 요지"로 인용
```
"""

from __future__ import annotations

from .base import Skill
from .loader import SkillRegistry, load_skills_from_dir

__all__ = ["Skill", "SkillRegistry", "load_skills_from_dir"]
