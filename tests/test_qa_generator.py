"""QA 생성기(teacher/qa_generator.py)의 단위 테스트입니다."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_factory.models import QAPair
from slm_factory.teacher.qa_generator import chunk_document


# ---------------------------------------------------------------------------
# QAGenerator를 생성하는 헬퍼 (create_teacher를 mock)
# ---------------------------------------------------------------------------


def _make_generator(mocker, make_config, **config_overrides):
    """create_teacher를 mock하여 QAGenerator 인스턴스를 생성하는 헬퍼입니다."""
    mock_teacher = MagicMock()
    mocker.patch(
        "slm_factory.teacher.create_teacher",
        return_value=mock_teacher,
    )
    from slm_factory.teacher.qa_generator import QAGenerator

    config = make_config(**config_overrides)
    gen = QAGenerator(config)
    return gen, mock_teacher


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """QAGenerator.build_prompt 메서드의 테스트입니다."""

    def test_기본_구조(self, mocker, make_config):
        """프롬프트에 system prompt, document, question 섹션이 포함되는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_prompt(
            doc_title="테스트 문서",
            content="문서 내용입니다.",
            question="무엇입니까?",
        )

        assert "시스템 지시사항" in prompt
        assert "테스트 문서" in prompt
        assert "무엇입니까?" in prompt
        assert "# 질문" in prompt

    def test_tables_포함_시_관련_표_존재(self, mocker, make_config):
        """tables 인자가 있으면 '관련 표' 섹션이 프롬프트에 포함되는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_prompt(
            doc_title="제목",
            content="내용",
            question="질문",
            tables=["| A | B |", "| C | D |"],
        )

        assert "관련 표" in prompt
        assert "| A | B |" in prompt
        assert "| C | D |" in prompt

    def test_tables_None_시_관련_표_없음(self, mocker, make_config):
        """tables=None이면 '관련 표' 섹션이 프롬프트에 포함되지 않는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_prompt(
            doc_title="제목",
            content="내용",
            question="질문",
            tables=None,
        )

        assert "관련 표" not in prompt

    def test_content_잘림_확인(self, mocker, make_config):
        """content가 max_context_chars보다 길면 잘리고 truncated 표시가 나타나는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config, teacher={"max_context_chars": 50})
        long_content = "A" * 100
        prompt = gen.build_prompt(
            doc_title="제목",
            content=long_content,
            question="질문",
        )

        assert "[이하 생략...]" in prompt
        # 원본 100자가 아닌 50자로 잘렸는지 확인
        assert "A" * 100 not in prompt


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """QAGenerator.parse_response 메서드의 테스트입니다."""

    def test_정상_JSON_instruction_output(self, mocker, make_config):
        """정상 JSON 응답을 올바르게 파싱하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        text = '{"instruction": "질문입니다", "output": "답변입니다"}'
        result = gen.parse_response(text)

        assert isinstance(result, dict)
        assert result["instruction"] == "질문입니다"
        assert result["output"] == "답변입니다"

    def test_question_answer_키_정규화(self, mocker, make_config):
        """'question'/'answer' 키가 'instruction'/'output'으로 정규화되는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        text = '{"question": "Q?", "answer": "A!"}'
        result = gen.parse_response(text)

        assert isinstance(result, dict)
        assert result["instruction"] == "Q?"
        assert result["output"] == "A!"

    def test_JSON_배열_첫_항목(self, mocker, make_config):
        """JSON 배열 응답에서 첫 번째 항목을 추출하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        text = '[{"instruction": "첫 번째", "output": "답변1"}]'
        result = gen.parse_response(text)

        assert isinstance(result, dict)
        assert result["instruction"] == "첫 번째"

    def test_data_래핑_풀림(self, mocker, make_config):
        """{"data": [...]} 래핑된 응답이 올바르게 풀리는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        text = '{"data": [{"instruction": "래핑", "output": "답변"}]}'
        result = gen.parse_response(text)

        assert isinstance(result, dict)
        assert result["instruction"] == "래핑"

    def test_코드_펜스_제거(self, mocker, make_config):
        """코드 펜스(```json ... ```)가 제거된 후 파싱되는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        text = '```json\n{"instruction": "펜스", "output": "제거됨"}\n```'
        result = gen.parse_response(text)

        assert isinstance(result, dict)
        assert result["instruction"] == "펜스"
        assert result["output"] == "제거됨"

    def test_잘못된_JSON_None_반환(self, mocker, make_config):
        """유효하지 않은 JSON 문자열에 대해 None을 반환하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        result = gen.parse_response("이것은 JSON이 아닙니다")
        assert result is None

    def test_필수_키_누락_None_반환(self, mocker, make_config):
        """'instruction'/'output' 키가 없는 JSON에 대해 None을 반환하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        result = gen.parse_response('{"foo": "bar"}')
        assert result is None


# ---------------------------------------------------------------------------
# generate_for_document
# ---------------------------------------------------------------------------


class TestGenerateForDocument:
    """QAGenerator.generate_for_document 메서드의 테스트입니다."""

    def test_teacher_generate_mock으로_QAPair_생성(
        self, mocker, make_config, make_parsed_doc
    ):
        """teacher.generate를 mock하여 QAPair를 올바르게 생성하는지 확인합니다."""
        gen, mock_teacher = _make_generator(
            mocker,
            make_config,
            questions={"categories": {"factual": ["문서의 주요 내용은?"]}},
        )
        mock_teacher.generate.return_value = json.dumps(
            {
                "instruction": "문서의 주요 내용은?",
                "output": "이것은 충분히 긴 답변입니다. 최소 길이를 충족합니다.",
            }
        )

        doc = make_parsed_doc()
        pairs = gen.generate_for_document(doc)

        assert len(pairs) == 1
        assert isinstance(pairs[0], QAPair)
        assert pairs[0].source_doc == doc.doc_id
        mock_teacher.generate.assert_called_once()

    def test_질문_없으면_빈_리스트(self, mocker, make_config, make_parsed_doc):
        """질문이 설정되지 않으면 빈 리스트를 반환하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        doc = make_parsed_doc()
        pairs = gen.generate_for_document(doc)
        assert pairs == []

    def test_파싱_실패_시_건너뜀(self, mocker, make_config, make_parsed_doc):
        """teacher.generate 응답 파싱이 실패해도 에러 없이 건너뛰는지 확인합니다."""
        gen, mock_teacher = _make_generator(
            mocker,
            make_config,
            questions={"categories": {"cat": ["질문1", "질문2"]}},
        )
        # 첫 번째는 잘못된 JSON, 두 번째는 정상
        mock_teacher.generate.side_effect = [
            "invalid json",
            '{"instruction": "질문2", "output": "충분히 긴 정상 답변입니다. 최소 길이를 충족합니다."}',
        ]

        doc = make_parsed_doc()
        pairs = gen.generate_for_document(doc)

        assert len(pairs) == 1
        assert pairs[0].question == "질문2"


# ---------------------------------------------------------------------------
# save_alpaca
# ---------------------------------------------------------------------------


class TestSaveAlpaca:
    """QAGenerator.save_alpaca 메서드의 테스트입니다."""

    def test_파일_저장_및_JSON_구조(self, mocker, make_config, make_qa_pair, tmp_path):
        """save_alpaca가 올바른 Alpaca 형식 JSON을 저장하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        pair = make_qa_pair()
        output_path = tmp_path / "output" / "qa.json"

        result_path = gen.save_alpaca([pair], output_path)

        assert result_path.exists()

        # 저장된 JSON 로드 및 구조 검증
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert "instruction" in data[0]
        assert "input" in data[0]
        assert "output" in data[0]
        assert data[0]["input"] == ""  # Alpaca 형식에서 input은 빈 문자열

    def test_여러_쌍_저장(self, mocker, make_config, make_qa_pair, tmp_path):
        """여러 QAPair를 저장하면 JSON 배열의 길이가 올바른지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        pairs = [make_qa_pair(question=f"Q{i}") for i in range(3)]
        output_path = tmp_path / "multi.json"

        gen.save_alpaca(pairs, output_path)

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 3


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocument:
    """chunk_document 함수의 테스트입니다."""

    def test_짧은_문서_단일_청크(self):
        """chunk_size보다 짧은 문서는 단일 청크로 반환되는지 확인합니다."""
        content = "짧은 문서입니다."
        chunks = chunk_document(content, chunk_size=10000, overlap=500)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_긴_문서_여러_청크(self):
        """chunk_size보다 긴 문서가 여러 청크로 분할되는지 확인합니다."""
        content = "A" * 30000
        chunks = chunk_document(content, chunk_size=10000, overlap=500)
        assert len(chunks) >= 3

    def test_중첩_영역_확인(self):
        """연속된 청크 사이에 중첩 영역이 존재하는지 확인합니다."""
        content = "X" * 5000
        chunks = chunk_document(content, chunk_size=2000, overlap=200)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            tail = chunks[i][-200:]
            assert tail in chunks[i + 1]

    def test_문단_경계_존중(self):
        """문단 경계(\\n\\n)에서 분할을 시도하는지 확인합니다."""
        para1 = "A" * 1500
        para2 = "B" * 1500
        content = para1 + "\n\n" + para2
        chunks = chunk_document(content, chunk_size=2000, overlap=100)
        assert len(chunks) >= 2
        assert chunks[0].endswith("A" * 10)

    def test_빈_문서(self):
        """빈 문자열이 단일 빈 청크로 반환되는지 확인합니다."""
        chunks = chunk_document("", chunk_size=10000, overlap=500)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_정확히_chunk_size(self):
        """문서 길이가 정확히 chunk_size일 때 단일 청크로 반환되는지 확인합니다."""
        content = "X" * 5000
        chunks = chunk_document(content, chunk_size=5000, overlap=500)
        assert len(chunks) == 1
        assert chunks[0] == content


# ---------------------------------------------------------------------------
# build_prompt chunk_info
# ---------------------------------------------------------------------------


class TestBuildPromptChunkInfo:
    """build_prompt의 chunk_info 파라미터 테스트입니다."""

    def test_chunk_info_포함_시_제목에_Part_표시(self, mocker, make_config):
        """chunk_info가 주어지면 제목에 Part 정보가 포함되는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_prompt(
            doc_title="테스트 문서",
            content="내용",
            question="질문",
            chunk_info="Part 2/5",
        )
        assert "테스트 문서 (Part 2/5)" in prompt

    def test_chunk_info_None_시_기본_제목(self, mocker, make_config):
        """chunk_info가 None이면 제목에 Part가 포함되지 않는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_prompt(
            doc_title="테스트 문서",
            content="내용",
            question="질문",
            chunk_info=None,
        )
        assert "Part" not in prompt
        assert "테스트 문서" in prompt


# ---------------------------------------------------------------------------
# _get_doc_chunks
# ---------------------------------------------------------------------------


class TestGetDocChunks:
    """QAGenerator._get_doc_chunks 메서드의 테스트입니다."""

    def test_청킹_비활성화_시_단일_청크(self, mocker, make_config, make_parsed_doc):
        """chunking.enabled=False이면 원본 문서를 (content, None)으로 반환하는지 확인합니다."""
        gen, _ = _make_generator(mocker, make_config)
        doc = make_parsed_doc(content="테스트 내용")
        chunks = gen._get_doc_chunks(doc)
        assert len(chunks) == 1
        assert chunks[0] == ("테스트 내용", None)

    def test_청킹_활성화_긴_문서(self, mocker, make_config, make_parsed_doc):
        """chunking이 활성화되고 문서가 길면 여러 (chunk, Part N/M) 튜플을 반환하는지 확인합니다."""
        gen, _ = _make_generator(
            mocker,
            make_config,
            chunking={"enabled": True, "chunk_size": 1000, "overlap_chars": 100},
        )
        doc = make_parsed_doc(content="X" * 3000)
        chunks = gen._get_doc_chunks(doc)
        assert len(chunks) >= 3
        assert chunks[0][1] == f"Part 1/{len(chunks)}"
        assert chunks[-1][1] == f"Part {len(chunks)}/{len(chunks)}"

    def test_청킹_활성화_짧은_문서(self, mocker, make_config, make_parsed_doc):
        """chunking이 활성화되어도 짧은 문서는 (content, None)으로 반환되는지 확인합니다."""
        gen, _ = _make_generator(
            mocker,
            make_config,
            chunking={"enabled": True, "chunk_size": 10000, "overlap_chars": 500},
        )
        doc = make_parsed_doc(content="짧은 문서")
        chunks = gen._get_doc_chunks(doc)
        assert len(chunks) == 1
        assert chunks[0][1] is None


# ---------------------------------------------------------------------------
# build_auto_generate_prompt
# ---------------------------------------------------------------------------


class TestBuildAutoGeneratePrompt:
    def test_기본_구조(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_auto_generate_prompt(
            doc_title="테스트 문서",
            content="문서 내용입니다.",
            num_questions=5,
        )

        assert "시스템 지시사항" in prompt
        assert "테스트 문서" in prompt
        assert "5개의 다양한 질문-답변 쌍을 생성하세요" in prompt
        assert "JSON 형식" in prompt

    def test_chunk_info_포함(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_auto_generate_prompt(
            doc_title="문서",
            content="내용",
            num_questions=3,
            chunk_info="Part 2/5",
        )

        assert "문서 (Part 2/5)" in prompt

    def test_tables_포함(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_auto_generate_prompt(
            doc_title="제목",
            content="내용",
            num_questions=3,
            tables=["| A | B |"],
        )

        assert "관련 표" in prompt
        assert "| A | B |" in prompt

    def test_content_잘림(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config, teacher={"max_context_chars": 50})
        prompt = gen.build_auto_generate_prompt(
            doc_title="제목",
            content="A" * 100,
            num_questions=3,
        )

        assert "[이하 생략...]" in prompt
        assert "A" * 100 not in prompt

    def test_질문_안내_포함(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        prompt = gen.build_auto_generate_prompt(
            doc_title="제목",
            content="내용",
            num_questions=10,
        )

        assert "사실 기반" in prompt
        assert "수치, 날짜, 이름" in prompt
        assert '"instruction"' in prompt
        assert '"output"' in prompt


# ---------------------------------------------------------------------------
# parse_auto_response
# ---------------------------------------------------------------------------


class TestParseAutoResponse:
    def test_정상_JSON_배열(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = json.dumps(
            [
                {"instruction": "질문1", "output": "답변1"},
                {"instruction": "질문2", "output": "답변2"},
            ],
            ensure_ascii=False,
        )
        results = gen.parse_auto_response(text)

        assert len(results) == 2
        assert results[0]["instruction"] == "질문1"
        assert results[1]["output"] == "답변2"

    def test_question_answer_키_정규화(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = json.dumps(
            [
                {"question": "Q?", "answer": "A!"},
            ]
        )
        results = gen.parse_auto_response(text)

        assert len(results) == 1
        assert results[0]["instruction"] == "Q?"
        assert results[0]["output"] == "A!"

    def test_빈_배열(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        results = gen.parse_auto_response("[]")
        assert results == []

    def test_잘못된_JSON_빈_리스트(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        results = gen.parse_auto_response("이것은 JSON이 아닙니다")
        assert results == []

    def test_코드_펜스_제거(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = '```json\n[{"instruction": "Q", "output": "A"}]\n```'
        results = gen.parse_auto_response(text)
        assert len(results) == 1

    def test_data_래핑_풀림(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = json.dumps(
            {
                "data": [
                    {"instruction": "Q1", "output": "A1"},
                ]
            }
        )
        results = gen.parse_auto_response(text)
        assert len(results) == 1
        assert results[0]["instruction"] == "Q1"

    def test_items_래핑_풀림(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = json.dumps(
            {
                "items": [
                    {"instruction": "Q1", "output": "A1"},
                ]
            }
        )
        results = gen.parse_auto_response(text)
        assert len(results) == 1

    def test_단일_객체_리스트_변환(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = '{"instruction": "Q", "output": "A"}'
        results = gen.parse_auto_response(text)
        assert len(results) == 1
        assert results[0]["instruction"] == "Q"

    def test_부분_파싱_필수_키_누락_건너뜀(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = json.dumps(
            [
                {"instruction": "유효", "output": "답변"},
                {"foo": "bar"},
                {"instruction": "유효2", "output": "답변2"},
            ],
            ensure_ascii=False,
        )
        results = gen.parse_auto_response(text)
        assert len(results) == 2
        assert results[0]["instruction"] == "유효"
        assert results[1]["instruction"] == "유효2"

    def test_비딕셔너리_항목_건너뜀(self, mocker, make_config):
        gen, _ = _make_generator(mocker, make_config)
        text = json.dumps(
            [
                {"instruction": "Q", "output": "A"},
                "문자열 항목",
                42,
            ]
        )
        results = gen.parse_auto_response(text)
        assert len(results) == 1
