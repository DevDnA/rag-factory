"""QA 쌍에 대한 규칙 기반 검증 필터."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ValidationConfig

from ..models import QAPair
from ..utils import get_logger

logger = get_logger("validator.rules")


@dataclass
class ValidationResult:
    """QA 쌍 검증 결과."""
    passed: bool
    reasons: list[str] = field(default_factory=list)


class RuleValidator:
    """설정 가능한 규칙을 사용하여 QA 쌍을 검증합니다.
    
    적용되는 규칙 (순서대로):
    1. 빈 값 확인: 질문 또는 답변이 비어있거나 공백이면 거부
    2. 길이 확인: 답변이 최소 길이보다 짧거나 최대 길이보다 길면 거부
    3. 거부 패턴: 답변이 정규식 패턴과 일치하면 거부 (예: "I don't know")
    4. 중복 제거: 중복된 질문-답변 쌍 거부
    """
    
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self._seen_pairs: set[str] = set()  # 중복 제거용 — (질문, 답변) 해시 저장
        self._compiled_patterns: list[re.Pattern] = [
            re.compile(p) for p in config.reject_patterns
        ]
    
    def validate_one(self, pair: QAPair) -> ValidationResult:
        """단일 QA 쌍을 검증합니다. 통과/실패 및 이유를 포함한 ValidationResult를 반환합니다."""
        reasons = []
        
        # 1. 빈 값 확인
        if self.config.remove_empty:
            if not pair.question.strip() or not pair.answer.strip():
                reasons.append("empty_question_or_answer")
                return ValidationResult(passed=False, reasons=reasons)
        
        # 2. 길이 확인
        answer_len = len(pair.answer.strip())
        if answer_len < self.config.min_answer_length:
            reasons.append(f"answer_too_short ({answer_len} < {self.config.min_answer_length})")
        if answer_len > self.config.max_answer_length:
            reasons.append(f"answer_too_long ({answer_len} > {self.config.max_answer_length})")
        
        # 3. 거부 패턴
        for pattern in self._compiled_patterns:
            if pattern.search(pair.answer):
                reasons.append(f"matched_reject_pattern: {pattern.pattern}")
        
        # 4. 중복 제거
        if self.config.deduplicate:
            pair_key = f"{pair.question.strip().lower()}|{pair.answer.strip().lower()}"
            if pair_key in self._seen_pairs:
                reasons.append("duplicate")
            else:
                self._seen_pairs.add(pair_key)
        
        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)
    
    def validate_batch(self, pairs: list[QAPair]) -> tuple[list[QAPair], list[tuple[QAPair, ValidationResult]]]:
        """QA 쌍 배치를 검증합니다.
        
        반환값:
            (수락된_쌍, 거부된_쌍_및_이유) 튜플
        """
        accepted = []
        rejected = []
        
        for pair in pairs:
            result = self.validate_one(pair)
            if result.passed:
                accepted.append(pair)
            else:
                rejected.append((pair, result))
        
        total = len(pairs)
        n_accepted = len(accepted)
        n_rejected = len(rejected)
        logger.info(
            f"Validation complete: {n_accepted}/{total} accepted, "
            f"{n_rejected}/{total} rejected"
        )
        
        # 거부 이유 요약 로깅
        if rejected:
            reason_counts: dict[str, int] = {}
            for _, result in rejected:
                for reason in result.reasons:
                    # 계산을 위해 이유 정규화 (세부사항 제거)
                    key = reason.split(":")[0].split("(")[0].strip()
                    reason_counts[key] = reason_counts.get(key, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                logger.info("  Rejection reason: %s (%d)", reason, count)
        
        return accepted, rejected
    
    def reset_dedup(self) -> None:
        """중복 제거 캐시를 초기화합니다 (예: 문서 배치 간)."""
        self._seen_pairs.clear()
