"""교사 모델(teacher) 모듈의 통합 테스트입니다.

OllamaTeacher, OpenAICompatTeacher, create_teacher 팩토리 함수를 검증합니다.
외부 네트워크 호출은 httpx mock으로 대체합니다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.config import TeacherConfig
from slm_factory.teacher.ollama import OllamaTeacher
from slm_factory.teacher.openai_compat import OpenAICompatTeacher
from slm_factory.teacher import create_teacher


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_teacher_config(**overrides) -> TeacherConfig:
    """테스트용 TeacherConfig를 간편하게 생성합니다."""
    defaults = {
        "backend": "ollama",
        "model": "qwen3.5:9b",
        "api_base": "http://localhost:11434",
        "api_key": None,
        "temperature": 0.3,
        "timeout": 300,
    }
    defaults.update(overrides)
    return TeacherConfig(**defaults)


# ---------------------------------------------------------------------------
# OllamaTeacher
# ---------------------------------------------------------------------------


class TestOllamaTeacherInit:
    """OllamaTeacher 초기화 테스트입니다."""

    def test_필드_설정(self):
        """__init__에서 model, api_base, temperature, timeout 필드가 올바르게 설정되는지 확인합니다."""
        config = _make_teacher_config(
            model="test-model",
            api_base="http://myhost:11434/",
            temperature=0.7,
            timeout=60,
        )
        teacher = OllamaTeacher(config)

        assert teacher.model == "test-model"
        assert teacher.api_base == "http://myhost:11434"  # 후행 슬래시 제거
        assert teacher.temperature == 0.7
        assert teacher.timeout == 60


class TestOllamaTeacherGenerate:
    """OllamaTeacher.generate 메서드의 테스트입니다."""

    def test_정상_응답(self, mocker):
        """httpx.post가 정상 응답을 반환할 때 텍스트를 올바르게 추출하는지 확인합니다."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "  test answer  "}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("httpx.post", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        result = teacher.generate("질문입니다")

        assert result == "test answer"
        httpx.post.assert_called_once()

    def test_타임아웃_런타임에러(self, mocker):
        """httpx.TimeoutException 발생 시 RuntimeError를 발생시키는지 확인합니다."""
        mocker.patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))

        teacher = OllamaTeacher(_make_teacher_config())
        with pytest.raises(RuntimeError):
            teacher.generate("질문입니다")

    def test_연결_에러_런타임에러(self, mocker):
        """httpx.ConnectError 발생 시 RuntimeError를 발생시키는지 확인합니다."""
        mocker.patch("httpx.post", side_effect=httpx.ConnectError("conn"))

        teacher = OllamaTeacher(_make_teacher_config())
        with pytest.raises(RuntimeError):
            teacher.generate("질문입니다")

    def test_HTTP_에러_런타임에러(self, mocker):
        """HTTP 상태 에러(500 등) 발생 시 RuntimeError를 발생시키는지 확인합니다."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
        mocker.patch("httpx.post", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        with pytest.raises(RuntimeError):
            teacher.generate("질문입니다")

    def test_빈_응답(self, mocker):
        """응답이 빈 문자열일 때 빈 문자열을 반환하는지 확인합니다."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": ""}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("httpx.post", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        result = teacher.generate("질문입니다")

        assert result == ""


class TestOllamaTeacherAgenerate:
    """OllamaTeacher.agenerate 비동기 메서드의 테스트입니다."""

    @pytest.mark.asyncio
    async def test_비동기_정상_응답(self, mocker):
        """httpx.AsyncClient.post가 정상 응답을 반환할 때 비동기로 텍스트를 추출하는지 확인합니다."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "async answer"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        teacher = OllamaTeacher(_make_teacher_config())
        result = await teacher.agenerate("질문입니다")

        assert result == "async answer"


class TestOllamaTeacherHealthCheck:
    """OllamaTeacher.health_check 메서드의 테스트입니다."""

    def test_정상_True(self, mocker):
        """API 서버가 정상 응답을 반환할 때 True를 반환하는지 확인합니다."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("httpx.get", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        assert teacher.health_check() is True

    def test_실패_False(self, mocker):
        """API 서버 연결 실패 시 False를 반환하는지 확인합니다."""
        mocker.patch("httpx.get", side_effect=httpx.ConnectError("connection refused"))

        teacher = OllamaTeacher(_make_teacher_config())
        assert teacher.health_check() is False


# ---------------------------------------------------------------------------
# OpenAICompatTeacher
# ---------------------------------------------------------------------------


class TestOpenAICompatTeacherInit:
    """OpenAICompatTeacher 초기화 테스트입니다."""

    def test_필드_설정(self):
        """__init__에서 model, api_base, api_key, temperature, timeout 필드가 올바르게 설정되는지 확인합니다."""
        config = _make_teacher_config(
            backend="openai",
            model="gpt-4",
            api_base="http://api.example.com/",
            api_key="sk-test-key",
            temperature=0.5,
            timeout=120,
        )
        teacher = OpenAICompatTeacher(config)

        assert teacher.model == "gpt-4"
        assert teacher.api_base == "http://api.example.com"
        assert teacher.api_key == "sk-test-key"
        assert teacher.temperature == 0.5
        assert teacher.timeout == 120


class TestOpenAICompatTeacherHeaders:
    """OpenAICompatTeacher._headers 메서드의 테스트입니다."""

    def test_API_키_있을_때_Authorization_포함(self):
        """api_key가 설정되어 있으면 Authorization 헤더가 포함되는지 확인합니다."""
        config = _make_teacher_config(backend="openai", api_key="sk-test-key")
        teacher = OpenAICompatTeacher(config)
        headers = teacher._headers()

        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_API_키_없을_때_Authorization_없음(self):
        """api_key가 None이면 Authorization 헤더가 포함되지 않는지 확인합니다."""
        config = _make_teacher_config(backend="openai", api_key=None)
        teacher = OpenAICompatTeacher(config)
        headers = teacher._headers()

        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"


class TestOpenAICompatTeacherGenerate:
    """OpenAICompatTeacher.generate 메서드의 테스트입니다."""

    def test_정상_응답(self, mocker):
        """httpx.post가 OpenAI 형식의 정상 응답을 반환할 때 텍스트를 올바르게 추출하는지 확인합니다."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "  openai answer  "}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("httpx.post", return_value=mock_resp)

        config = _make_teacher_config(backend="openai")
        teacher = OpenAICompatTeacher(config)
        result = teacher.generate("질문입니다")

        assert result == "openai answer"


# ---------------------------------------------------------------------------
# create_teacher 팩토리 함수
# ---------------------------------------------------------------------------


class TestCreateTeacher:
    """create_teacher 팩토리 함수의 테스트입니다."""

    def test_ollama_백엔드(self):
        """backend='ollama'일 때 OllamaTeacher 인스턴스를 반환하는지 확인합니다."""
        config = _make_teacher_config(backend="ollama")
        teacher = create_teacher(config)
        assert isinstance(teacher, OllamaTeacher)

    def test_openai_백엔드(self):
        """backend='openai'일 때 OpenAICompatTeacher 인스턴스를 반환하는지 확인합니다."""
        config = _make_teacher_config(backend="openai")
        teacher = create_teacher(config)
        assert isinstance(teacher, OpenAICompatTeacher)

    def test_알_수_없는_백엔드_ValueError(self):
        """알 수 없는 backend 값을 지정하면 ValueError를 발생시키는지 확인합니다."""
        config = _make_teacher_config(backend="ollama")
        config.backend = "unknown"
        with pytest.raises(ValueError, match="Unknown teacher backend"):
            create_teacher(config)


# ---------------------------------------------------------------------------
# Ollama 스트리밍
# ---------------------------------------------------------------------------


class TestOllamaTeacherStreamGenerate:
    """OllamaTeacher.generate(stream=True)의 동기 스트리밍 테스트입니다."""

    def test_스트리밍_정상_응답(self, mocker):
        """NDJSON 스트리밍 응답에서 토큰을 정상적으로 누적합니다."""
        lines = [
            '{"response": "안녕", "done": false}',
            '{"response": "하세요", "done": false}',
            '{"response": "", "done": true}',
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mocker.patch("httpx.stream", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        result = teacher.generate("질문입니다", stream=True)

        assert result == "안녕하세요"

    def test_스트리밍_빈_응답(self, mocker):
        """스트리밍에서 빈 응답을 처리합니다."""
        lines = ['{"response": "", "done": true}']

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mocker.patch("httpx.stream", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        result = teacher.generate("질문입니다", stream=True)

        assert result == ""

    def test_스트리밍_타임아웃_RuntimeError(self, mocker):
        """스트리밍 중 타임아웃 시 RuntimeError를 발생시킵니다."""
        mocker.patch("httpx.stream", side_effect=httpx.TimeoutException("timeout"))
        mocker.patch("slm_factory.teacher.ollama.time.sleep")

        teacher = OllamaTeacher(_make_teacher_config())
        with pytest.raises(RuntimeError, match="타임아웃"):
            teacher.generate("질문입니다", stream=True)

    def test_스트리밍_잘못된_JSON_건너뜀(self, mocker):
        """유효하지 않은 JSON 라인을 건너뛰고 정상 토큰만 수집합니다."""
        lines = [
            '{"response": "정상", "done": false}',
            "invalid json line",
            '{"response": "토큰", "done": false}',
            '{"response": "", "done": true}',
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mocker.patch("httpx.stream", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        result = teacher.generate("질문입니다", stream=True)

        assert result == "정상토큰"

    def test_stream_False_기존_동작_유지(self, mocker):
        """stream=False(기본값)일 때 기존 httpx.post 경로를 사용합니다."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "기존 응답"}
        mock_resp.raise_for_status = MagicMock()
        mocker.patch("httpx.post", return_value=mock_resp)

        teacher = OllamaTeacher(_make_teacher_config())
        result = teacher.generate("질문입니다")

        assert result == "기존 응답"
        httpx.post.assert_called_once()


class TestOllamaTeacherStreamAgenerate:
    """OllamaTeacher.agenerate(stream=True)의 비동기 스트리밍 테스트입니다."""

    @pytest.mark.asyncio
    async def test_비동기_스트리밍_정상_응답(self, mocker):
        """비동기 NDJSON 스트리밍에서 토큰을 정상적으로 누적합니다."""
        lines = [
            '{"response": "비동기", "done": false}',
            '{"response": " 스트리밍", "done": false}',
            '{"response": "", "done": true}',
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = mock_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        teacher = OllamaTeacher(_make_teacher_config())
        result = await teacher.agenerate("질문입니다", stream=True)

        assert result == "비동기 스트리밍"
