"""자동 캘리브레이션(calibration.py)의 단위 테스트입니다."""

import pytest

from slm_factory.calibration import (
    _find_section_boundaries,
    auto_chunk_size,
    auto_num_epochs,
    auto_questions_per_chunk,
    section_aware_chunk,
)


class TestAutoChunkSize:
    def test_작은_문서_ceiling_반환(self):
        content = "짧은 문서" * 100
        result = auto_chunk_size(content, max_context_chars=12000)
        assert result == 10000

    def test_큰_문서_ceiling_미만(self):
        content = "가" * 100_000
        result = auto_chunk_size(content, max_context_chars=12000)
        assert 2000 <= result <= 10000

    def test_ceiling_제한_준수(self):
        content = "나" * 500_000
        result = auto_chunk_size(content, max_context_chars=12000)
        assert result <= 10000

    def test_최소값_2000_보장(self):
        content = "다" * 50_000
        result = auto_chunk_size(content, max_context_chars=4500)
        assert result >= 2000

    def test_조밀_단락_작은_청크(self):
        short_paras = "\n\n".join(["짧은 단락입니다. 정보가 밀집." for _ in range(200)])
        dense_result = auto_chunk_size(short_paras, max_context_chars=12000)

        long_paras = "\n\n".join(
            ["이것은 매우 긴 서술형 문단입니다. " * 20 for _ in range(20)]
        )
        narrative_result = auto_chunk_size(long_paras, max_context_chars=12000)

        assert dense_result <= narrative_result

    def test_서술_단락_큰_청크(self):
        long_paras = "\n\n".join(
            [
                "서술형으로 길게 작성된 문단입니다. 많은 설명을 포함합니다. " * 15
                for _ in range(30)
            ]
        )
        result = auto_chunk_size(long_paras, max_context_chars=12000)
        assert result >= 2000

    def test_빈_단락_기본_밀도(self):
        content = "가" * 20000
        result = auto_chunk_size(content, max_context_chars=12000)
        assert 2000 <= result <= 10000

    def test_max_context_변경(self):
        content = "라" * 50_000
        small_ctx = auto_chunk_size(content, max_context_chars=6000)
        large_ctx = auto_chunk_size(content, max_context_chars=20000)
        assert small_ctx <= large_ctx


class TestAutoQuestionsPerChunk:
    def test_최소값_3_보장(self):
        result = auto_questions_per_chunk("짧은 청크")
        assert result == 3

    def test_최대값_15_제한(self):
        massive_chunk = "이것은 긴 문장입니다. " * 5000
        result = auto_questions_per_chunk(massive_chunk)
        assert result <= 15

    def test_숫자_보너스(self):
        no_nums = "가나다라마바사" * 200
        with_nums = (
            no_nums + "\n2024년 매출 1000억원 증가율 15.3% 직원 500명 예산 200조"
        )
        result_no = auto_questions_per_chunk(no_nums)
        result_with = auto_questions_per_chunk(with_nums)
        assert result_with >= result_no

    def test_목록_보너스(self):
        plain = "일반 텍스트입니다. " * 200
        with_list = plain + "\n".join([f"- 항목 {i}" for i in range(15)])
        result_plain = auto_questions_per_chunk(plain)
        result_list = auto_questions_per_chunk(with_list)
        assert result_list >= result_plain

    def test_한국어_정부문서_패턴(self):
        gov_doc = (
            "제1조(목적) 이 법은 대한민국의 공공기관 운영에 관한 사항을 규정함을 목적으로 한다.\n"
            "1) 공공기관의 정의\n"
            "2) 운영 원칙\n"
            "가) 투명성 원칙\n"
            "나) 효율성 원칙\n"
            "2024년 예산: 500조원\n"
            "전년 대비 증가율: 5.2%\n"
            "소속 공무원: 100만명\n"
        ) * 10
        result = auto_questions_per_chunk(gov_doc)
        assert result >= 3

    def test_1200자당_1개_기본비율(self):
        content = "가" * 6000
        result = auto_questions_per_chunk(content)
        assert result == 5

    def test_괄호_번호_목록_인식(self):
        chunk = "내용입니다. " * 100 + "\n".join(
            [f"({i}) 항목 설명" for i in range(1, 11)]
        )
        result = auto_questions_per_chunk(chunk)
        assert result >= 3


class TestAutoConfigIntegration:
    def test_chunk_size_auto_config(self):
        from slm_factory.config import ChunkingConfig

        cfg = ChunkingConfig(chunk_size="auto")
        assert cfg.chunk_size == "auto"

    def test_questions_per_chunk_auto_config(self):
        from slm_factory.config import QuestionsConfig

        cfg = QuestionsConfig(questions_per_chunk="auto")
        assert cfg.questions_per_chunk == "auto"

    def test_정수값_하위호환(self):
        from slm_factory.config import ChunkingConfig, QuestionsConfig

        c_cfg = ChunkingConfig(chunk_size=8000)
        assert c_cfg.chunk_size == 8000

        q_cfg = QuestionsConfig(questions_per_chunk=5)
        assert q_cfg.questions_per_chunk == 5

    def test_동시_auto_설정(self):
        from slm_factory.config import SLMConfig

        cfg = SLMConfig(
            chunking={"chunk_size": "auto"},
            questions={"questions_per_chunk": "auto"},
        )
        assert cfg.chunking.chunk_size == "auto"
        assert cfg.questions.questions_per_chunk == "auto"

    def test_기본값_auto(self):
        from slm_factory.config import ChunkingConfig, QuestionsConfig

        assert ChunkingConfig().chunk_size == "auto"
        assert QuestionsConfig().questions_per_chunk == "auto"


class TestFindSectionBoundaries:
    def test_로마_숫자_헤더_감지(self):
        content = (
            "서문 텍스트\n"
            "Ⅰ. 사업 개요\n내용1\n"
            "Ⅱ. 추진 배경\n내용2\n"
            "Ⅲ. 사업 범위\n내용3\n"
        )
        boundaries = _find_section_boundaries(content)
        assert len(boundaries) == 3
        assert all(
            content[b] == "Ⅰ" or content[b] == "Ⅱ" or content[b] == "Ⅲ"
            for b in boundaries
        )

    def test_아라비아_숫자_헤더_감지(self):
        content = "1. 개요\n내용A\n2. 목적\n내용B\n3. 범위\n내용C\n"
        boundaries = _find_section_boundaries(content)
        assert len(boundaries) == 3

    def test_연도_제외(self):
        content = "2026. 연도는 헤더가 아닙니다\n" * 5
        boundaries = _find_section_boundaries(content)
        assert len(boundaries) == 0

    def test_헤더_없으면_빈_리스트(self):
        content = "일반 텍스트입니다. 헤더 패턴이 없습니다.\n" * 10
        boundaries = _find_section_boundaries(content)
        assert boundaries == []

    def test_3개_미만이면_빈_리스트(self):
        content = "Ⅰ. 사업 개요\n내용1\nⅡ. 추진 배경\n내용2\n"
        boundaries = _find_section_boundaries(content)
        assert boundaries == []


class TestSectionAwareChunk:
    def _make_section(self, header: str, body_chars: int = 600) -> str:
        return f"{header}\n" + "가" * body_chars

    def test_기본_섹션_분할(self):
        sections = [
            self._make_section("Ⅰ. 개요", 800),
            self._make_section("Ⅱ. 배경", 800),
            self._make_section("Ⅲ. 범위", 800),
        ]
        content = "\n".join(sections)
        chunks = section_aware_chunk(content, max_chunk_size=5000)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 5000

    def test_소형_섹션_병합(self):
        sections = [
            self._make_section("Ⅰ. 개요", 200),
            self._make_section("Ⅱ. 배경", 200),
            self._make_section("Ⅲ. 범위", 200),
            self._make_section("Ⅳ. 기타", 200),
        ]
        content = "\n".join(sections)
        chunks = section_aware_chunk(content, max_chunk_size=5000)
        total_sections_chars = sum(len(s) for s in sections)
        assert len(chunks) < 4
        assert sum(len(c) for c in chunks) > 0

    def test_대형_섹션_분할(self):
        large_body = "\n\n".join(["긴 문단입니다. " * 50 for _ in range(20)])
        sections = [
            f"Ⅰ. 개요\n{large_body}",
            self._make_section("Ⅱ. 배경", 600),
            self._make_section("Ⅲ. 범위", 600),
        ]
        content = "\n".join(sections)
        chunks = section_aware_chunk(content, max_chunk_size=2000)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 2000

    def test_200자_미만_필터링(self):
        sections = [
            self._make_section("Ⅰ. 개요", 800),
            "Ⅱ. 목차",
            "Ⅲ. 참고",
            self._make_section("Ⅳ. 본문", 800),
        ]
        content = "\n".join(sections)
        chunks = section_aware_chunk(content, max_chunk_size=50000)
        for chunk in chunks:
            assert len(chunk.strip()) >= 200

    def test_헤더_없으면_폴백(self):
        content = "\n\n".join(["일반 텍스트 문단입니다. " * 30 for _ in range(10)])
        chunks = section_aware_chunk(content, max_chunk_size=2000)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 2000

    def test_한국어_RFP_패턴(self):
        content = (
            "제안요청서\n\n"
            "Ⅰ. 사업 개요\n" + "사업 개요 내용입니다. " * 100 + "\n\n"
            "Ⅱ. 현황 및 문제점\n" + "현황 분석 내용입니다. " * 200 + "\n\n"
            "Ⅲ. 사업 추진 방향\n" + "추진 방향 내용입니다. " * 50 + "\n\n"
            "Ⅳ. 제안 요청 사항\n" + "제안 요청 상세 내용입니다. " * 300 + "\n\n"
            "Ⅴ. 제안서 작성 안내\n" + "작성 안내 내용입니다. " * 80 + "\n\n"
            "Ⅵ. 평가 기준\n" + "평가 기준 내용입니다. " * 60 + "\n"
        )
        chunks = section_aware_chunk(content, max_chunk_size=5000)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 5000
            assert len(chunk.strip()) >= 200

    def test_하위_호환_정수_chunk_size(self):
        from slm_factory.teacher.qa_generator import chunk_document

        content = "가나다라마바사" * 500
        expected = chunk_document(content, 1000, 200)
        assert len(expected) >= 1


class TestAutoNumEpochs:
    def test_소량_데이터_3에포크(self):
        assert auto_num_epochs(50) == 3

    def test_100미만_3에포크(self):
        assert auto_num_epochs(99) == 3

    def test_100_2에포크(self):
        assert auto_num_epochs(100) == 2

    def test_500미만_2에포크(self):
        assert auto_num_epochs(499) == 2

    def test_500이상_1에포크(self):
        assert auto_num_epochs(500) == 1

    def test_대량_데이터_1에포크(self):
        assert auto_num_epochs(10000) == 1

    def test_경계값_0(self):
        assert auto_num_epochs(0) == 3
