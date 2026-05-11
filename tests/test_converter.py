"""변환기(converter.py) 모듈의 통합 테스트입니다.

ChatFormatter 클래스의 메시지 빌드, 형식화, 배치 처리, 파일 저장 기능을 검증합니다.
transformers 토크나이저는 mock으로 대체합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_factory.converter import ChatFormatter
from slm_factory.models import QAPair


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_formatter(make_config, system_prompt="도움이 되는 어시스턴트입니다."):
    """테스트용 ChatFormatter를 생성하고 토크나이저를 mock으로 설정합니다."""
    config = make_config(
        student={"model": "test-model", "max_seq_length": 512},
        questions={"system_prompt": system_prompt},
    )
    formatter = ChatFormatter(config)

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<formatted text>"
    mock_tokenizer.encode.return_value = [1, 2, 3]  # 토큰 수 = 3
    formatter._tokenizer = mock_tokenizer

    return formatter, mock_tokenizer


# ---------------------------------------------------------------------------
# ChatFormatter.__init__
# ---------------------------------------------------------------------------


class TestChatFormatterInit:
    """ChatFormatter 초기화 테스트입니다."""

    def test_필드_설정(self, make_config):
        """model_name, max_seq_length, system_prompt 필드가 올바르게 설정되는지 확인합니다."""
        config = make_config(
            student={"model": "my-model", "max_seq_length": 2048},
            questions={"system_prompt": "테스트 프롬프트"},
        )
        formatter = ChatFormatter(config)

        assert formatter.model_name == "my-model"
        assert formatter.max_seq_length == 2048
        assert formatter.system_prompt == "테스트 프롬프트"
        assert formatter._tokenizer is None


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """ChatFormatter.build_messages 메서드의 테스트입니다."""

    def test_시스템_프롬프트_있을_때_3개_메시지(self, make_config, make_qa_pair):
        """시스템 프롬프트가 있으면 system, user, assistant 3개 메시지를 생성하는지 확인합니다."""
        formatter, _ = _make_formatter(make_config, system_prompt="시스템 메시지")
        pair = make_qa_pair(question="테스트 질문?", answer="테스트 답변입니다.")

        messages = formatter.build_messages(pair)

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "시스템 메시지"}
        assert messages[1] == {"role": "user", "content": "테스트 질문?"}
        assert messages[2] == {"role": "assistant", "content": "테스트 답변입니다."}

    def test_시스템_프롬프트_없을_때_2개_메시지(self, make_config, make_qa_pair):
        """시스템 프롬프트가 빈 문자열이면 user, assistant 2개 메시지만 생성하는지 확인합니다."""
        formatter, _ = _make_formatter(make_config, system_prompt="")
        pair = make_qa_pair(question="Q?", answer="A.")

        messages = formatter.build_messages(pair)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# format_one
# ---------------------------------------------------------------------------


class TestFormatOne:
    """ChatFormatter.format_one 메서드의 테스트입니다."""

    def test_정상_형식화(self, make_config, make_qa_pair):
        """토크나이저가 정상적으로 채팅 템플릿을 적용하는 경우를 확인합니다."""
        formatter, mock_tok = _make_formatter(make_config)
        mock_tok.apply_chat_template.return_value = "<formatted>"
        pair = make_qa_pair()

        result = formatter.format_one(pair)

        assert result == "<formatted>"
        mock_tok.apply_chat_template.assert_called()

    def test_시스템_역할_실패_시_폴백(self, make_config, make_qa_pair):
        """첫 번째 apply_chat_template 호출이 실패하고 두 번째 호출이 성공하는 폴백 경로를 확인합니다."""
        formatter, mock_tok = _make_formatter(make_config)
        # 첫 번째 호출: 시스템 역할 포함 → 예외, 두 번째 호출: 시스템 역할 없이 → 성공
        mock_tok.apply_chat_template.side_effect = [
            Exception("system role not supported"),
            "<fallback formatted>",
        ]
        pair = make_qa_pair()

        result = formatter.format_one(pair)

        assert result == "<fallback formatted>"
        assert mock_tok.apply_chat_template.call_count == 2


# ---------------------------------------------------------------------------
# format_batch
# ---------------------------------------------------------------------------


class TestFormatBatch:
    """ChatFormatter.format_batch 메서드의 테스트입니다."""

    def test_정상_쌍과_초과_쌍_필터링(self, make_config, make_qa_pair):
        """max_seq_length를 초과하는 쌍이 필터링되는지 확인합니다."""
        formatter, mock_tok = _make_formatter(make_config)
        # max_seq_length=512, 토큰 수는 batch encode 결과의 길이
        # 정상 쌍: 3 토큰, 초과 쌍: 1000 토큰
        short_tokens = list(range(3))
        long_tokens = list(range(1000))
        mock_tok.apply_chat_template.return_value = "<formatted>"
        # format_batch는 self.tokenizer(texts, ...) 배치 호출을 사용
        mock_tok.return_value = {"input_ids": [short_tokens, long_tokens, short_tokens]}

        pairs = [
            make_qa_pair(question="짧은 질문1"),
            make_qa_pair(question="매우 긴 질문"),
            make_qa_pair(question="짧은 질문2"),
        ]

        results = formatter.format_batch(pairs)

        # 1000 토큰 쌍은 512 초과이므로 필터링
        assert len(results) == 2

# ---------------------------------------------------------------------------
# save_training_data
# ---------------------------------------------------------------------------


class TestSaveTrainingData:
    """ChatFormatter.save_training_data 메서드의 테스트입니다."""

    def test_JSONL_파일_생성(self, make_config, make_qa_pair, tmp_path):
        """JSONL 파일이 올바르게 생성되고 내용이 정확한지 확인합니다."""
        formatter, mock_tok = _make_formatter(make_config)
        mock_tok.apply_chat_template.return_value = "<formatted>"
        # format_batch는 self.tokenizer(texts, ...) 배치 호출을 사용
        mock_tok.return_value = {"input_ids": [[1, 2, 3], [1, 2, 3]]}

        pairs = [make_qa_pair(question="Q1"), make_qa_pair(question="Q2")]
        output_path = tmp_path / "train.jsonl"

        result = formatter.save_training_data(pairs, output_path)

        assert result.exists()
        lines = result.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "text" in data

# ---------------------------------------------------------------------------
# format_from_alpaca_file
# ---------------------------------------------------------------------------


class TestFormatFromAlpacaFile:
    """ChatFormatter.format_from_alpaca_file 메서드의 테스트입니다."""

    def test_Alpaca_JSON_변환(self, make_config, tmp_path):
        """Alpaca JSON 파일을 JSONL로 올바르게 변환하는지 확인합니다."""
        formatter, mock_tok = _make_formatter(make_config)
        mock_tok.apply_chat_template.return_value = "<formatted>"
        # format_batch는 self.tokenizer(texts, ...) 배치 호출을 사용
        mock_tok.return_value = {"input_ids": [[1, 2, 3], [1, 2, 3]]}

        # Alpaca 형식 입력 파일 생성
        alpaca_data = [
            {
                "instruction": "질문1",
                "input": "",
                "output": "이것은 충분히 긴 테스트 답변입니다. 최소 길이를 충족시키기 위한 내용입니다.",
            },
            {
                "instruction": "질문2",
                "input": "",
                "output": "이것도 충분히 긴 테스트 답변입니다. 최소 길이를 충족시키기 위한 내용을 추가합니다.",
            },
        ]
        input_path = tmp_path / "qa_alpaca.json"
        input_path.write_text(json.dumps(alpaca_data, ensure_ascii=False), encoding="utf-8")

        output_path = tmp_path / "training_data.jsonl"
        result = formatter.format_from_alpaca_file(input_path, output_path)

        assert result.exists()
        lines = result.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
