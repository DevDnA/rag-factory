"""ReAct 텍스트 출력 파서입니다.

LLM이 생성한 자유 텍스트에서 Thought / Action / Final Answer를
정규식으로 추출합니다. 파싱 실패 시 전체 텍스트를 Final Answer로
처리하는 graceful degradation을 지원합니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ...utils import get_logger

logger = get_logger("rag.agent.parser")

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------


@dataclass
class ParsedStep:
    """ReAct 한 스텝의 파싱 결과."""

    thought: str = ""
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None
    raw: str = ""


# ---------------------------------------------------------------------------
# 파서 정규식 — 영문·한국어 레이블 동시 지원
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(
    r"(?:Thought|생각)\s*:\s*(.+?)(?=\n(?:Action|행동|Final Answer|최종 답변)\s*:|$)",
    re.DOTALL | re.IGNORECASE,
)

_ACTION_RE = re.compile(
    r"(?:Action|행동)\s*:\s*(\S+)",
    re.IGNORECASE,
)

_ACTION_INPUT_RE = re.compile(
    r"(?:Action Input|행동 입력)\s*:\s*(.+?)(?=\n(?:Thought|생각|Observation|관찰|Final Answer|최종 답변)\s*:|$)",
    re.DOTALL | re.IGNORECASE,
)

_FINAL_ANSWER_RE = re.compile(
    r"(?:Final Answer|최종 답변)\s*:\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_REASONING_ARTIFACT_RE = re.compile(
    r"\n\s*(?:Thought|생각|Action|행동|Action Input|행동 입력|Observation|관찰)\s*:.*",
    re.DOTALL | re.IGNORECASE,
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Gemma 등 모델이 출력하는 특수 태그 (허용 목록 기반)
_SPECIAL_TAG_RE = re.compile(
    r"</?(?:end_of_turn|start_of_turn|channel|pad|eos|bos|sep|cls|mask|unk)\|?>",
    re.IGNORECASE,
)


def _clean_final_answer(text: str) -> str:
    """Final Answer에서 추론 잔여물, thinking/특수 태그를 제거합니다."""
    text = _THINK_TAG_RE.sub("", text)
    text = _SPECIAL_TAG_RE.sub("", text)
    text = _REASONING_ARTIFACT_RE.sub("", text)
    return text.strip()


def parse_react_output(text: str) -> ParsedStep:
    """LLM 출력 텍스트를 ``ParsedStep``으로 파싱합니다.

    파싱에 실패하면 전체 텍스트를 ``final_answer``로 반환합니다.
    """
    step = ParsedStep(raw=text)

    # Final Answer 우선 검사
    m_final = _FINAL_ANSWER_RE.search(text)
    if m_final:
        step.final_answer = _clean_final_answer(m_final.group(1).strip())

    # Thought 추출
    m_thought = _THOUGHT_RE.search(text)
    if m_thought:
        step.thought = m_thought.group(1).strip()

    # Action + Action Input 추출
    m_action = _ACTION_RE.search(text)
    if m_action and not m_final:
        step.action = m_action.group(1).strip().lower()

        m_input = _ACTION_INPUT_RE.search(text)
        if m_input:
            raw_input = m_input.group(1).strip()
            step.action_input = _parse_action_input(raw_input)
        else:
            step.action_input = {}

    # Fallback: 아무 패턴도 매칭 안 되면 전체를 답변으로 처리.
    # 단, 마커가 전혀 없고 텍스트가 매우 짧으면(<10자) 의미 있는 답변일 가능성이
    # 낮으므로 final_answer를 채우지 않고 force_answer 경로로 escalate되도록 둠.
    if not step.thought and not step.action and not step.final_answer:
        cleaned = text.strip()
        if not cleaned:
            return step
        has_marker = bool(
            re.search(
                r"(?:Action|행동|Final Answer|최종 답변|Thought|생각)\s*:",
                cleaned,
                re.IGNORECASE,
            )
        )
        if not has_marker and len(cleaned) < 10:
            logger.debug(
                "ReAct 파싱 실패 — 너무 짧고 마커도 없어 force_answer로 escalate"
            )
            return step
        logger.debug("ReAct 파싱 실패 — 전체 텍스트를 답변으로 처리")
        step.final_answer = cleaned

    return step


def _parse_action_input(raw: str) -> dict:
    """Action Input 문자열을 dict로 파싱합니다.

    JSON 파싱을 시도하고, 실패 시 ``{"query": raw}``로 폴백합니다.
    """
    # JSON 객체 추출 시도
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        json_str = raw[brace_start : brace_end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 단순 문자열이면 query로 감싸기
    cleaned = raw.strip().strip('"').strip("'")
    if cleaned:
        return {"query": cleaned}
    return {}
