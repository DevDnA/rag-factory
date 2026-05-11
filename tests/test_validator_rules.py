"""규칙 기반 검증(validator/rules.py)의 단위 테스트입니다."""

from slm_factory.config import ValidationConfig
from slm_factory.models import QAPair
from slm_factory.validator.rules import RuleValidator, ValidationResult


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    """ValidationResult 데이터클래스의 테스트입니다."""

    def test_기본값(self):
        """ValidationResult의 기본값(passed=True, reasons=[])을 검증합니다."""
        result = ValidationResult(passed=True)
        assert result.passed is True
        assert result.reasons == []

    def test_실패_결과(self):
        """실패 결과에 이유가 포함되는지 확인합니다."""
        result = ValidationResult(passed=False, reasons=["test_reason"])
        assert result.passed is False
        assert "test_reason" in result.reasons


# ---------------------------------------------------------------------------
# RuleValidator.validate_one
# ---------------------------------------------------------------------------


class TestValidateOne:
    """RuleValidator.validate_one 메서드의 테스트입니다."""

    def test_정상_쌍_통과(self, make_qa_pair, sample_validation_config):
        """유효한 QA 쌍이 검증을 통과하는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = make_qa_pair()
        result = validator.validate_one(pair)
        assert result.passed is True
        assert result.reasons == []

    def test_빈_질문_거부(self, sample_validation_config):
        """빈 질문이 포함된 쌍이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(question="", answer="충분히 긴 답변 내용입니다. 최소 길이를 충족합니다.")
        result = validator.validate_one(pair)
        assert result.passed is False
        assert "empty_question_or_answer" in result.reasons

    def test_빈_답변_거부(self, sample_validation_config):
        """빈 답변이 포함된 쌍이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(question="질문입니다", answer="")
        result = validator.validate_one(pair)
        assert result.passed is False
        assert "empty_question_or_answer" in result.reasons

    def test_짧은_답변_거부(self, sample_validation_config):
        """min_answer_length보다 짧은 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(question="질문", answer="짧음")  # 2자 < 20자
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("answer_too_short" in r for r in result.reasons)

    def test_긴_답변_거부(self, sample_validation_config):
        """max_answer_length보다 긴 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(question="질문", answer="A" * 2001)
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("answer_too_long" in r for r in result.reasons)

    def test_i_dont_know_패턴_거부(self, sample_validation_config):
        """'I don't know' 패턴이 포함된 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(
            question="질문",
            answer="I don't know the answer to this question at all honestly.",
        )
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("matched_reject_pattern" in r for r in result.reasons)

    def test_not_available_패턴_거부(self, sample_validation_config):
        """'not available' 패턴이 포함된 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(
            question="질문",
            answer="This information is not available in the provided document content.",
        )
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("matched_reject_pattern" in r for r in result.reasons)

    def test_document_does_not_contain_패턴_거부(self, sample_validation_config):
        """'the document does not contain' 패턴이 포함된 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(
            question="질문",
            answer="The document does not contain information about this specific topic.",
        )
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("matched_reject_pattern" in r for r in result.reasons)

    def test_한국어_알수없_패턴_거부(self, sample_validation_config):
        """한국어 '알 수 없' 패턴이 포함된 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(
            question="질문",
            answer="해당 내용은 주어진 문서에서 확인할 수 없습니다.",
        )
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("matched_reject_pattern" in r for r in result.reasons)

    def test_한국어_정보없음_패턴_거부(self, sample_validation_config):
        """한국어 '정보가 없' 패턴이 포함된 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(
            question="질문",
            answer="이에 대한 정보가 없습니다.",
        )
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("matched_reject_pattern" in r for r in result.reasons)

    def test_한국어_문서에서_찾을수없_패턴_거부(self, sample_validation_config):
        """한국어 '문서에서 찾을 수 없' 패턴이 포함된 답변이 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = QAPair(
            question="질문",
            answer="관련 내용은 문서에서 찾을 수 없습니다.",
        )
        result = validator.validate_one(pair)
        assert result.passed is False
        assert any("matched_reject_pattern" in r for r in result.reasons)

    def test_중복_쌍_거부(self, make_qa_pair, sample_validation_config):
        """동일한 질문-답변 쌍이 두 번째부터 중복으로 거부되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = make_qa_pair()

        result1 = validator.validate_one(pair)
        assert result1.passed is True

        result2 = validator.validate_one(pair)
        assert result2.passed is False
        assert "duplicate" in result2.reasons

    def test_remove_empty_False일_때_빈_값_통과(self):
        """remove_empty=False로 설정하면 빈 값이 통과하는지 확인합니다."""
        config = ValidationConfig(
            remove_empty=False,
            min_answer_length=0,
            reject_patterns=[],
        )
        validator = RuleValidator(config)
        pair = QAPair(question="", answer="")
        result = validator.validate_one(pair)
        assert result.passed is True

    def test_deduplicate_False일_때_중복_통과(self, make_qa_pair):
        """deduplicate=False로 설정하면 중복이 통과하는지 확인합니다."""
        config = ValidationConfig(deduplicate=False)
        validator = RuleValidator(config)
        pair = make_qa_pair()

        result1 = validator.validate_one(pair)
        assert result1.passed is True

        result2 = validator.validate_one(pair)
        assert result2.passed is True


# ---------------------------------------------------------------------------
# RuleValidator.validate_batch
# ---------------------------------------------------------------------------


class TestValidateBatch:
    """RuleValidator.validate_batch 메서드의 테스트입니다."""

    def test_혼합_쌍_분리(self, make_qa_pair, sample_validation_config):
        """유효한 쌍과 유효하지 않은 쌍이 올바르게 분리되는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pairs = [
            make_qa_pair(question="Q1"),
            QAPair(question="", answer=""),  # 빈 값 → 거부
            make_qa_pair(question="Q2"),
        ]

        accepted, rejected = validator.validate_batch(pairs)
        assert len(accepted) == 2
        assert len(rejected) == 1

    def test_빈_리스트(self, sample_validation_config):
        """빈 리스트를 전달하면 빈 결과를 반환하는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        accepted, rejected = validator.validate_batch([])
        assert accepted == []
        assert rejected == []


# ---------------------------------------------------------------------------
# RuleValidator.reset_dedup
# ---------------------------------------------------------------------------


class TestResetDedup:
    """RuleValidator.reset_dedup 메서드의 테스트입니다."""

    def test_리셋_후_동일_쌍_재통과(self, make_qa_pair, sample_validation_config):
        """reset_dedup 호출 후 동일한 쌍이 다시 통과하는지 확인합니다."""
        validator = RuleValidator(sample_validation_config)
        pair = make_qa_pair()

        # 첫 번째 통과
        result1 = validator.validate_one(pair)
        assert result1.passed is True

        # 두 번째 → 중복 거부
        result2 = validator.validate_one(pair)
        assert result2.passed is False

        # 리셋 후 다시 통과
        validator.reset_dedup()
        result3 = validator.validate_one(pair)
        assert result3.passed is True
