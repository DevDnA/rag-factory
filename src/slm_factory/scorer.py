"""교사 LLM을 사용한 QA 쌍 품질 점수 평가."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from .config import ScoringConfig, TeacherConfig
    from .teacher.base import BaseTeacher

from .models import QAPair
from .utils import get_logger, run_bounded

logger = get_logger("scorer")


class QualityScorer:
    """교사 LLM을 사용하여 QA 쌍의 품질을 1~5점으로 평가합니다."""

    def __init__(
        self, teacher: BaseTeacher, config: ScoringConfig, teacher_config: TeacherConfig
    ) -> None:
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config

    def _build_scoring_prompt(self, pair: QAPair, source_text: str = "") -> str:
        """점수 평가를 위한 프롬프트를 구성합니다.

        매개변수
        ----------
        pair:
            평가할 QA 쌍입니다.
        source_text:
            원본 문서 텍스트입니다. 제공되면 답변의 근거성(환각 여부)도 평가합니다.
        """
        source_section = ""
        faithfulness_criteria = ""
        if source_text:
            # 원본 문서가 너무 길면 앞부분만 사용 (프롬프트 크기 제한)
            truncated = source_text[:3000]
            source_section = f"## 원본 문서 (발췌)\n{truncated}\n\n"
            faithfulness_criteria = (
                "- 답변이 원본 문서의 내용에 근거하는지 (환각/지어낸 정보가 없는지)\n"
            )

        return (
            "다음 질문-답변 쌍의 품질을 1~5점으로 평가해주세요.\n\n"
            f"{source_section}"
            "## 평가 기준\n"
            "- 1점: 답변이 완전히 잘못되었거나 무관합니다\n"
            "- 2점: 답변이 부분적으로 맞지만 심각한 오류가 있습니다\n"
            "- 3점: 답변이 대체로 맞지만 불완전하거나 부정확한 부분이 있습니다\n"
            "- 4점: 답변이 정확하고 충분하지만 약간의 개선 여지가 있습니다\n"
            "- 5점: 답변이 정확하고 완전하며 잘 구조화되어 있습니다\n\n"
            "## 핵심 체크리스트\n"
            "- 답변이 질문에 직접적으로 응답하는지\n"
            "- 구체적 수치, 날짜, 이름 등 세부 정보가 포함되어 있는지\n"
            f"{faithfulness_criteria}"
            "\n"
            "## 평가 대상\n"
            f"질문: {pair.question}\n"
            f"답변: {pair.answer}\n\n"
            "반드시 아래 JSON 형식으로만 응답하세요:\n"
            '{"score": <1-5 정수>, "reason": "<평가 근거 한 문장>"}'
        )

    def _parse_score(self, text: str) -> tuple[int, str] | None:
        """LLM 응답에서 점수와 이유를 추출합니다."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            score = int(data.get("score", 0))
            reason = str(data.get("reason", ""))
            if 1 <= score <= 5:
                return score, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("점수 JSON 파싱 실패, 정규식 fallback 시도: %s", text[:80])

        # 패턴 1: 'score' 또는 '점수' 키워드 뒤의 숫자
        match = re.search(r"(?:score|점수)\D{0,10}([1-5])", text, re.IGNORECASE)
        if match:
            return int(match.group(1)), "점수만 추출됨"

        # 패턴 2: 독립된 1~5 숫자가 정확히 하나만 존재할 때 (word boundary 사용)
        digits = re.findall(r"\b([1-5])\b", text)
        if len(digits) == 1:
            return int(digits[0]), "점수만 추출됨"

        logger.warning("점수 파싱 실패: %s", text[:100])
        return None

    async def score_one(
        self,
        pair: QAPair,
        source_text: str = "",
    ) -> tuple[QAPair, int, str]:
        """단일 QA 쌍을 점수 평가합니다."""
        prompt = self._build_scoring_prompt(pair, source_text=source_text)

        kwargs: dict[str, Any] = {}
        if self.teacher_config.backend == "ollama":
            kwargs["format"] = "json"
            kwargs["think"] = False

        response = await self.teacher.agenerate(prompt, **kwargs)
        result = self._parse_score(response)

        if result is None:
            return pair, 0, "점수 파싱 실패 — 기본값 0 적용 (필터링 대상)"

        score, reason = result
        return pair, score, reason

    def _build_batch_scoring_prompt(
        self,
        pairs: list[QAPair],
        source_text: str = "",
    ) -> str:
        source_section = ""
        if source_text:
            truncated = source_text[:3000]
            source_section = f"## 원본 문서 (발췌)\n{truncated}\n\n"

        qa_list = ""
        for i, pair in enumerate(pairs):
            qa_list += f"[{i}] 질문: {pair.question}\n    답변: {pair.answer}\n"

        return (
            f"다음 {len(pairs)}개 질문-답변 쌍의 품질을 각각 1~5점으로 평가해주세요.\n\n"
            f"{source_section}"
            "## 평가 기준\n"
            "- 1점: 완전히 잘못됨  - 3점: 대체로 맞지만 불완전  - 5점: 정확하고 완전함\n\n"
            f"## 평가 대상\n{qa_list}\n"
            '반드시 JSON 형식으로 응답: {"scores": [{"id": 0, "score": 점수, "reason": "이유"}, ...]}'
        )

    def _parse_batch_scores(
        self, text: str, count: int
    ) -> list[tuple[int, str] | None]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        results: list[tuple[int, str] | None] = [None] * count
        try:
            data = json.loads(text)
            items = data.get("scores", data.get("items", []))
            if isinstance(data, list):
                items = data
            for item in items:
                idx = int(item.get("id", -1))
                score = int(item.get("score", 0))
                reason = str(item.get("reason", ""))
                if 0 <= idx < count and 1 <= score <= 5:
                    results[idx] = (score, reason)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("배치 점수 JSON 파싱 실패, 개별 fallback: %s", text[:100])
        return results

    async def score_batch(
        self,
        pairs: list[QAPair],
        source_text: str = "",
    ) -> list[tuple[QAPair, int, str]]:
        prompt = self._build_batch_scoring_prompt(pairs, source_text)
        kwargs: dict[str, Any] = {}
        if self.teacher_config.backend == "ollama":
            kwargs["format"] = "json"
            kwargs["think"] = False
        response = await self.teacher.agenerate(prompt, **kwargs)
        parsed = self._parse_batch_scores(response, len(pairs))

        results = []
        for i, pair in enumerate(pairs):
            item = parsed[i]
            if item is not None:
                results.append((pair, item[0], item[1]))
            else:
                results.append((pair, 0, "배치 파싱 실패"))
        return results

    async def score_all(
        self,
        pairs: list[QAPair],
        source_texts: dict[str, str] | None = None,
    ) -> tuple[list[QAPair], list[tuple[QAPair, int, str]]]:
        source_map = source_texts or {}
        batch_size = 10
        accepted: list[QAPair] = []
        filtered: list[tuple[QAPair, int, str]] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        with progress:
            task_id = progress.add_task("품질 점수 평가 중...", total=len(batches))

            for batch in batches:
                source = source_map.get(batch[0].source_doc, "") if batch else ""
                try:
                    results = await self.score_batch(batch, source)
                    for pair, score, reason in results:
                        if score >= self.config.threshold:
                            accepted.append(pair)
                        else:
                            filtered.append((pair, score, reason))
                except Exception as e:
                    logger.error("배치 점수 평가 실패: %s", e)
                    for pair in batch:
                        filtered.append((pair, 0, str(e)))
                progress.update(task_id, advance=1)

        logger.info(
            "품질 점수 평가 완료: %d/%d 통과 (threshold=%.1f), %d 제거",
            len(accepted),
            len(pairs),
            self.config.threshold,
            len(filtered),
        )
        return accepted, filtered
