"""PPT 파서(parsers/ppt.py)의 단위 테스트입니다."""

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.modules["olefile"] = MagicMock()

from slm_factory.parsers.ppt import PPTParser


def _build_ppt_stream(texts: list[str]) -> bytes:
    """TextCharsAtom 레코드를 포함하는 최소 PPT 스트림 바이트를 구성합니다.

    각 텍스트를 UTF-16LE TextCharsAtom(recType=0x0FA0) 레코드로 인코딩합니다.
    recVer=0 → 리프 레코드이므로 파서가 헤더 + recLen만큼 전진합니다.
    """
    _RT_TEXT_CHARS_ATOM = 0x0FA0
    buf = bytearray()
    for text in texts:
        payload = text.encode("utf-16-le")
        # recVer(4bit)=0, recInstance(12bit)=0 → 0x0000
        header = struct.pack("<HHI", 0x0000, _RT_TEXT_CHARS_ATOM, len(payload))
        buf += header + payload
    return bytes(buf)


def _build_ppt_stream_ansi(texts: list[str]) -> bytes:
    """TextBytesAtom 레코드를 포함하는 최소 PPT 스트림 바이트를 구성합니다."""
    _RT_TEXT_BYTES_ATOM = 0x0FA8
    buf = bytearray()
    for text in texts:
        payload = text.encode("cp1252", errors="replace")
        header = struct.pack("<HHI", 0x0000, _RT_TEXT_BYTES_ATOM, len(payload))
        buf += header + payload
    return bytes(buf)


class TestPPTParser:
    """PPTParser 클래스의 테스트입니다."""

    def test_can_parse_ppt_True(self):
        """.ppt 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = PPTParser()
        assert parser.can_parse(Path("presentation.ppt")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = PPTParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_FileNotFoundError_when_file_missing(self, tmp_path):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = PPTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.ppt")

    def test_parse_RuntimeError_when_olefile_not_installed(self, tmp_path):
        """olefile이 설치되지 않았을 때 RuntimeError를 발생시키는지 확인합니다."""
        ppt_file = tmp_path / "test.ppt"
        ppt_file.write_bytes(b"dummy")

        parser = PPTParser()

        with patch("slm_factory.parsers.ppt.HAS_OLEFILE", False):
            with pytest.raises(RuntimeError) as exc_info:
                parser.parse(ppt_file)

            assert "olefile" in str(exc_info.value)
            assert "uv sync --extra hwp" in str(exc_info.value)

    def test_parse_RuntimeError_when_invalid_ole(self, tmp_path):
        """유효한 OLE 파일이 아닐 때 RuntimeError를 발생시키는지 확인합니다."""
        ppt_file = tmp_path / "test.ppt"
        ppt_file.write_bytes(b"not an ole file")

        parser = PPTParser()

        with patch("slm_factory.parsers.ppt.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.ppt.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = False
                with pytest.raises(RuntimeError) as exc_info:
                    parser.parse(ppt_file)

                assert "OLE2" in str(exc_info.value) or "PPT" in str(exc_info.value)

    def test_parse_기본_PPT(self, tmp_path):
        """기본 PPT 파일을 파싱하여 한국어 텍스트를 추출하는지 확인합니다."""
        ppt_file = tmp_path / "presentation.ppt"
        ppt_file.write_bytes(b"dummy")

        slide_texts = ["첫 번째 단락입니다", "두 번째 슬라이드 내용"]
        stream_bytes = _build_ppt_stream(slide_texts)

        mock_stream = MagicMock()
        mock_stream.read.return_value = stream_bytes

        mock_ole = MagicMock()
        mock_ole.openstream.return_value = mock_stream
        mock_ole.get_metadata.side_effect = Exception("no metadata")

        parser = PPTParser()

        with patch("slm_factory.parsers.ppt.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.ppt.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = True
                mock_ole_mod.OleFileIO.return_value = mock_ole

                doc = parser.parse(ppt_file)

                assert doc.doc_id == "presentation.ppt"
                assert isinstance(doc.content, str)
                assert "첫 번째 단락" in doc.content
                assert "두 번째 슬라이드" in doc.content
                assert doc.title == "presentation"

    def test_parse_빈_PPT(self, tmp_path):
        """슬라이드가 없는 PPT 파일을 파싱하면 빈 content가 반환되는지 확인합니다."""
        ppt_file = tmp_path / "empty.ppt"
        ppt_file.write_bytes(b"dummy")

        # 텍스트 레코드 없는 빈 스트림
        stream_bytes = b""

        mock_stream = MagicMock()
        mock_stream.read.return_value = stream_bytes

        mock_ole = MagicMock()
        mock_ole.openstream.return_value = mock_stream
        mock_ole.get_metadata.side_effect = Exception("no metadata")

        parser = PPTParser()

        with patch("slm_factory.parsers.ppt.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.ppt.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = True
                mock_ole_mod.OleFileIO.return_value = mock_ole

                doc = parser.parse(ppt_file)

                assert doc.doc_id == "empty.ppt"
                assert doc.content == ""

    def test_parse_메타데이터_추출(self, tmp_path):
        """PPT 파일의 OLE 메타데이터에서 author와 title을 추출하는지 확인합니다."""
        from datetime import datetime

        ppt_file = tmp_path / "report.ppt"
        ppt_file.write_bytes(b"dummy")

        slide_texts = ["테스트 슬라이드 내용입니다."]
        stream_bytes = _build_ppt_stream(slide_texts)

        mock_stream = MagicMock()
        mock_stream.read.return_value = stream_bytes

        mock_meta = MagicMock()
        mock_meta.author = "홍길동"
        mock_meta.title = "테스트 제목"
        mock_meta.create_time = datetime(2024, 3, 15)

        mock_ole = MagicMock()
        mock_ole.openstream.return_value = mock_stream
        mock_ole.get_metadata.return_value = mock_meta

        parser = PPTParser()

        with patch("slm_factory.parsers.ppt.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.ppt.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = True
                mock_ole_mod.OleFileIO.return_value = mock_ole

                doc = parser.parse(ppt_file)

                assert doc.metadata.get("author") == "홍길동"
                assert doc.title == "테스트 제목"
                assert doc.metadata.get("date") == "2024-03-15"
