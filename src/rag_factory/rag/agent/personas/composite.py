"""다중 페르소나 합성 (Phase 14).

``CompositePersona``는 두 :class:`Persona` 인스턴스를 받아 명시적 우선순위로
속성을 합성합니다. IntentClassifier confidence가 낮거나 hybrid intent 신호
(예: "비교 + 왜")가 있을 때, primary persona의 합성 프롬프트에 secondary
persona의 핵심 지침을 명시 마커로 부착합니다.

설계 결정
---------
- ``allowed_tools``는 union — 두 페르소나가 필요로 하는 도구를 모두 허용해야
  multi-step plan이 깨지지 않습니다. 단, 둘 중 하나가 ``None`` (무제한)이면
  결과도 ``None``.
- ``synthesis_prompt_template``은 primary의 template을 그대로 두고, secondary의
  template에서 *핵심 규칙 섹션*만 추출해 명시된 마커 (``## 추가 분석 지침``)
  로 prepend된 후 합성됩니다. ``{history}`` / ``{context}`` / ``{query}``
  placeholder는 primary template에서 단 한 번만 등장해야 합니다 (``.format()``
  안전성).
- ``plan_strategy_hint``는 primary 우선 — secondary는 보조이므로 plan 전략
  자체는 primary가 결정.
"""

from __future__ import annotations

import re

from .base import Persona


# secondary template에서 추출할 핵심 규칙 라인 수.
_SECONDARY_INSTRUCTION_LINES = 4


class CompositePersona(Persona):
    """두 페르소나의 합성 — primary가 base, secondary가 보조 지침을 제공.

    Parameters
    ----------
    primary:
        주 페르소나 — name·전략 힌트·기본 prompt 구조를 결정.
    secondary:
        보조 페르소나 — synthesis_prompt_template에서 핵심 규칙만 추출되어
        primary의 template 본문에 부착됩니다.
    """

    def __init__(self, primary: Persona, secondary: Persona) -> None:
        self._primary = primary
        self._secondary = secondary

        # name·description은 합성 표시
        self.name = f"{primary.name}+{secondary.name}"
        self.description = (
            f"composite persona: {primary.name} (primary) + "
            f"{secondary.name} (secondary)"
        )

        # allowed_tools는 union — 둘 중 하나라도 None이면 None.
        if primary.allowed_tools is None or secondary.allowed_tools is None:
            self.allowed_tools = None
        else:
            self.allowed_tools = frozenset(
                primary.allowed_tools | secondary.allowed_tools
            )

        # plan 전략은 primary 우선.
        self.plan_strategy_hint = primary.plan_strategy_hint

        # synthesis prompt는 primary 본문에 secondary의 핵심 지침을 부착.
        self.synthesis_prompt_template = self._compose_template(
            primary.synthesis_prompt_template,
            secondary.synthesis_prompt_template,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _compose_template(
        primary_template: str | None,
        secondary_template: str | None,
    ) -> str | None:
        """primary template에 secondary의 핵심 규칙을 명시적 마커로 부착.

        - ``primary_template``이 ``None``이면 결과도 ``None`` (orchestrator가
          기본 ``ANSWER_SYNTHESIS_PROMPT`` fallback).
        - placeholder 충돌 방지: secondary에서 추출한 라인 안에
          ``{history}`` / ``{context}`` / ``{query}``가 포함되면 escape.
        - secondary가 ``None``이면 primary 그대로 반환.
        """
        if primary_template is None:
            return None
        if secondary_template is None:
            return primary_template

        secondary_excerpt = CompositePersona._extract_secondary_rules(
            secondary_template
        )
        if not secondary_excerpt:
            return primary_template

        marker = "\n\n## 추가 분석 지침 (secondary)\n"
        addendum = marker + secondary_excerpt + "\n"

        # primary template은 보통 `{history}\n[참고 문서]\n{context}\n\n질문: {query}\n답변:`
        # 형태로 끝납니다. 그 헤더 직전에 addendum을 삽입해야 합니다.
        # 안전한 삽입점: 첫 번째 `{history}` placeholder 직전.
        history_idx = primary_template.find("{history}")
        if history_idx == -1:
            # placeholder가 없으면 단순 prepend로 fallback (이 경우 plan 경로에서
            # 사용되지 않을 가능성 높음, 안전한 default).
            return addendum + primary_template

        composed = (
            primary_template[:history_idx]
            + addendum
            + primary_template[history_idx:]
        )
        return composed

    @staticmethod
    def _extract_secondary_rules(template: str) -> str:
        """secondary template에서 *핵심 규칙* 라인을 추출합니다.

        Researcher·Comparator·Analyst·Procedural 4개 persona는 모두 ``규칙:``
        섹션을 포함합니다. 그 섹션의 처음 N개 의미 있는 라인만 추출합니다.
        ``규칙:`` 마커가 없으면 template의 처음 N개 의미 라인을 fallback으로
        사용합니다.

        Placeholder (``{history}``, ``{context}``, ``{query}``)가 포함된 라인은
        그대로 두면 ``.format()`` 호출이 깨지므로 escape (``{{`` / ``}}``)
        처리합니다.
        """
        # `규칙:` 또는 `답변 구조:` 섹션 찾기
        marker_match = re.search(r"^(규칙:|답변 구조[^\n]*:)$", template, re.MULTILINE)
        start = marker_match.end() if marker_match else 0
        section = template[start:]

        lines: list[str] = []
        for raw_line in section.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            # placeholder 또는 다음 섹션 헤더(`[참고 문서]`, `질문:`, `{history}` 등)에서 중단
            if (
                line.startswith("{")
                or line.startswith("[참고")
                or line.startswith("질문:")
                or line.startswith("답변:")
            ):
                break
            # 다른 ## 헤더가 나오면 섹션 종료
            if line.startswith("##"):
                break
            # placeholder 안전 escape (단일 중괄호만)
            safe = line.replace("{", "{{").replace("}", "}}")
            lines.append(safe)
            if len(lines) >= _SECONDARY_INSTRUCTION_LINES:
                break

        return "\n".join(lines)


__all__ = ["CompositePersona"]
