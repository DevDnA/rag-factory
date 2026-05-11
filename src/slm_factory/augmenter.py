"""교사 LLM을 사용한 QA 쌍 데이터 증강 — 질문 패러프레이즈."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from .config import AugmentConfig, TeacherConfig
    from .teacher.base import BaseTeacher

from .models import QAPair
from .utils import get_logger, run_bounded

logger = get_logger("augmenter")


def _char_bigrams(text: str) -> set[tuple[str, str]]:
    """텍스트에서 공백 제거 후 문자 bigram 집합을 생성합니다."""
    t = text.replace(" ", "").lower()
    return {(t[i], t[i + 1]) for i in range(len(t) - 1)}


def _token_overlap_ratio(original: str, paraphrased: str) -> float:
    """두 문장의 문자 bigram 겹침 비율을 계산합니다 (0.0~1.0)."""
    orig_bg = _char_bigrams(original)
    para_bg = _char_bigrams(paraphrased)
    if not orig_bg or not para_bg:
        return 0.0
    intersection = orig_bg & para_bg
    return len(intersection) / max(len(orig_bg), len(para_bg))


class DataAugmenter:
    """교사 LLM을 사용하여 질문을 패러프레이즈하여 데이터를 증강합니다."""

    def __init__(self, teacher: BaseTeacher, config: AugmentConfig, teacher_config: TeacherConfig) -> None:
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config

    def _build_paraphrase_prompt(self, question: str, num_variants: int) -> str:
        """패러프레이즈 프롬프트를 구성합니다."""
        return (
            f"아래 질문을 의미를 유지하면서 {num_variants}개의 다른 표현으로 바꿔주세요.\n\n"
            f"원본 질문: {question}\n\n"
            "규칙:\n"
            "- 원래 질문과 동일한 의미를 유지할 것\n"
            "- 각 변형은 서로 다른 문장 구조나 어휘를 사용할 것\n"
            "- 자연스러운 한국어/영어로 작성할 것 (원본 언어를 따라갈 것)\n"
            "- 질문 형식을 유지할 것\n\n"
            '반드시 아래 JSON 형식으로만 응답하세요:\n'
            '{"questions": ["변형 질문 1", "변형 질문 2", ...]}'
        )

    def _parse_paraphrases(self, text: str) -> list[str]:
        """LLM 응답에서 패러프레이즈된 질문을 추출합니다."""
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "questions" in data:
                questions = data["questions"]
                if isinstance(questions, list):
                    return [q.strip() for q in questions if isinstance(q, str) and q.strip()]
            if isinstance(data, list):
                return [q.strip() for q in data if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError:
            logger.debug("패러프레이즈 JSON 파싱 실패: %s", text[:80])

        logger.warning("패러프레이즈 파싱 실패: %s", text[:100])
        return []

    async def paraphrase_one(self, pair: QAPair) -> list[QAPair]:
        """단일 QA 쌍의 질문을 패러프레이즈하여 증강된 QA 쌍을 생성합니다."""
        prompt = self._build_paraphrase_prompt(pair.question, self.config.num_variants)

        try:
            kwargs: dict[str, Any] = {}
            if self.teacher_config.backend == "ollama":
                kwargs["format"] = "json"
                kwargs["think"] = False

            response = await self.teacher.agenerate(prompt, **kwargs)
            paraphrases = self._parse_paraphrases(response)
        except Exception as e:
            logger.error("패러프레이즈 실패 (Q=%s...): %s", pair.question[:40], e)
            return []

        augmented_pairs = []
        min_sim = self.config.min_similarity
        for new_question in paraphrases[:self.config.num_variants]:
            overlap = _token_overlap_ratio(pair.question, new_question)
            if overlap < min_sim:
                logger.debug(
                    "패러프레이즈 제거 (겹침 %.2f < %.2f): '%s' → '%s'",
                    overlap, min_sim, pair.question[:30], new_question[:30],
                )
                continue
            augmented = QAPair(
                question=new_question,
                answer=pair.answer,
                instruction=new_question,
                source_doc=pair.source_doc,
                category=pair.category,
                context=pair.context,
                is_augmented=True,
            )
            augmented_pairs.append(augmented)

        return augmented_pairs

    async def augment_all(self, pairs: list[QAPair]) -> list[QAPair]:
        """전체 QA 쌍을 증강합니다. 원본 + 증강 쌍을 반환합니다."""
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        original_pairs = [p for p in pairs if not p.is_augmented]
        augmented: list[QAPair] = []

        with progress:
            task_id = progress.add_task("데이터 증강 중...", total=len(original_pairs))

            tasks = [
                run_bounded(semaphore, self.paraphrase_one(pair), progress, task_id)
                for pair in original_pairs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.error("증강 태스크 실패: %s", result)
                continue
            augmented.extend(result)

        total = len(pairs) + len(augmented)
        logger.info(
            "데이터 증강 완료: 원본 %d + 증강 %d = 총 %d개",
            len(pairs), len(augmented), total,
        )

        return list(pairs) + augmented
