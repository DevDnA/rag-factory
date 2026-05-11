"""HWP 바이너리 파서 테스트 — olefile 기반 HWP5 텍스트 추출을 검증합니다.

olefile을 mock하여 실제 HWP 파일 없이 파서 로직을 테스트합니다.
"""

from __future__ import annotations

import struct
import sys
import zlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import slm_factory.parsers.hwp as _hwp_mod

if not hasattr(_hwp_mod, "olefile"):
    _hwp_mod.olefile = MagicMock()  # type: ignore[attr-defined]


def _build_hwptag_para_text(text: str) -> bytes:
    """HWPTAG_PARA_TEXT (tag_id=67) 레코드를 바이너리로 생성합니다."""
    encoded = text.encode("utf-16-le")
    tag_id = 67
    level = 0
    size = len(encoded)
    if size < 0xFFF:
        header = tag_id | (level << 10) | (size << 20)
        return struct.pack("<I", header) + encoded
    header = tag_id | (level << 10) | (0xFFF << 20)
    return struct.pack("<I", header) + struct.pack("<I", size) + encoded


def _build_bodytext_data(*texts: str) -> bytes:
    """여러 텍스트 단락으로 BodyText 바이너리 데이터를 구성합니다."""
    parts = []
    for t in texts:
        parts.append(_build_hwptag_para_text(t))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# HWPParser 기본 동작 테스트
# ---------------------------------------------------------------------------


class TestHWPParserBasic:
    """HWPParser의 기본 파싱 기능을 검증합니다."""

    def test_파일_없으면_FileNotFoundError(self, tmp_path: Path):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시킵니다."""
        from slm_factory.parsers.hwp import HWPParser

        parser = HWPParser()
        with pytest.raises(FileNotFoundError, match="HWP not found"):
            parser.parse(tmp_path / "missing.hwp")

    def test_olefile_미설치시_RuntimeError(self, tmp_path: Path):
        """olefile이 없을 때 RuntimeError를 발생시킵니다."""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"dummy")

        with patch("slm_factory.parsers.hwp.HAS_OLEFILE", False):
            from slm_factory.parsers.hwp import HWPParser

            parser = HWPParser()
            with pytest.raises(RuntimeError, match="olefile이 필요합니다"):
                parser.parse(hwp_file)

    def test_비OLE_파일_RuntimeError(self, tmp_path: Path):
        """유효하지 않은 OLE2 파일에 대해 RuntimeError를 발생시킵니다."""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"this is not an OLE file")

        with (
            patch("slm_factory.parsers.hwp.HAS_OLEFILE", True),
            patch("slm_factory.parsers.hwp.olefile") as mock_olefile,
        ):
            mock_olefile.isOleFile.return_value = False

            from slm_factory.parsers.hwp import HWPParser

            parser = HWPParser()
            with pytest.raises(RuntimeError, match="유효한 HWP"):
                parser.parse(hwp_file)

    def test_정상_텍스트_추출(self, tmp_path: Path):
        """BodyText/Section0에서 텍스트가 정상적으로 추출되는지 확인합니다."""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"dummy")

        body_data = _build_bodytext_data("첫 번째 단락입니다.", "두 번째 단락입니다.")

        mock_ole = MagicMock()
        mock_ole.listdir.return_value = [["BodyText", "Section0"]]

        header_data = bytearray(40)
        header_data[36] = 0
        mock_ole.openstream.side_effect = lambda name: (
            BytesIO(bytes(header_data)) if name == "FileHeader" else BytesIO(body_data)
        )
        mock_ole.exists.return_value = False

        with (
            patch("slm_factory.parsers.hwp.HAS_OLEFILE", True),
            patch("slm_factory.parsers.hwp.olefile") as mock_olefile,
        ):
            mock_olefile.isOleFile.return_value = True
            mock_olefile.OleFileIO.return_value = mock_ole

            from slm_factory.parsers.hwp import HWPParser

            parser = HWPParser()
            result = parser.parse(hwp_file)

        assert "첫 번째 단락" in result.content
        assert "두 번째 단락" in result.content
        assert result.doc_id == "test.hwp"

    def test_압축_섹션_추출(self, tmp_path: Path):
        """압축된 BodyText 스트림에서 텍스트를 추출합니다."""
        hwp_file = tmp_path / "compressed.hwp"
        hwp_file.write_bytes(b"dummy")

        body_data = _build_bodytext_data("압축된 텍스트입니다.")
        compressed = zlib.compress(body_data)[2:-4]

        mock_ole = MagicMock()
        mock_ole.listdir.return_value = [["BodyText", "Section0"]]

        header_data = bytearray(40)
        header_data[36] = 1
        mock_ole.openstream.side_effect = lambda name: (
            BytesIO(bytes(header_data)) if name == "FileHeader" else BytesIO(compressed)
        )
        mock_ole.exists.return_value = False

        with (
            patch("slm_factory.parsers.hwp.HAS_OLEFILE", True),
            patch("slm_factory.parsers.hwp.olefile") as mock_olefile,
        ):
            mock_olefile.isOleFile.return_value = True
            mock_olefile.OleFileIO.return_value = mock_ole

            from slm_factory.parsers.hwp import HWPParser

            parser = HWPParser()
            result = parser.parse(hwp_file)

        assert "압축된 텍스트" in result.content


class TestHWPParserMultiSection:
    """HWPParser의 멀티 섹션 처리를 검증합니다."""

    def test_여러_섹션_텍스트_결합(self, tmp_path: Path):
        """여러 Section 스트림의 텍스트가 모두 결합되는지 확인합니다."""
        hwp_file = tmp_path / "multi.hwp"
        hwp_file.write_bytes(b"dummy")

        section0 = _build_bodytext_data("섹션0 텍스트")
        section1 = _build_bodytext_data("섹션1 텍스트")

        mock_ole = MagicMock()
        mock_ole.listdir.return_value = [
            ["BodyText", "Section0"],
            ["BodyText", "Section1"],
        ]

        header_data = bytearray(40)
        header_data[36] = 0

        def open_stream(name):
            if name == "FileHeader":
                return BytesIO(bytes(header_data))
            if name == ["BodyText", "Section0"]:
                return BytesIO(section0)
            return BytesIO(section1)

        mock_ole.openstream.side_effect = open_stream
        mock_ole.exists.return_value = False

        with (
            patch("slm_factory.parsers.hwp.HAS_OLEFILE", True),
            patch("slm_factory.parsers.hwp.olefile") as mock_olefile,
        ):
            mock_olefile.isOleFile.return_value = True
            mock_olefile.OleFileIO.return_value = mock_ole

            from slm_factory.parsers.hwp import HWPParser

            parser = HWPParser()
            result = parser.parse(hwp_file)

        assert "섹션0 텍스트" in result.content
        assert "섹션1 텍스트" in result.content


class TestHWPParserMetadata:
    """HWPParser의 메타데이터 추출을 검증합니다."""

    def test_파일명에서_날짜_추출(self, tmp_path: Path):
        """파일명의 YYMMDD 패턴에서 날짜가 추출되는지 확인합니다."""
        hwp_file = tmp_path / "report_240315.hwp"
        hwp_file.write_bytes(b"dummy")

        body_data = _build_bodytext_data("테스트")

        mock_ole = MagicMock()
        mock_ole.listdir.return_value = [["BodyText", "Section0"]]

        header_data = bytearray(40)
        mock_ole.openstream.side_effect = lambda name: (
            BytesIO(bytes(header_data)) if name == "FileHeader" else BytesIO(body_data)
        )
        mock_ole.exists.return_value = False

        with (
            patch("slm_factory.parsers.hwp.HAS_OLEFILE", True),
            patch("slm_factory.parsers.hwp.olefile") as mock_olefile,
        ):
            mock_olefile.isOleFile.return_value = True
            mock_olefile.OleFileIO.return_value = mock_ole

            from slm_factory.parsers.hwp import HWPParser

            parser = HWPParser()
            result = parser.parse(hwp_file)

        assert result.metadata.get("date") == "2024-03-15"


class TestHWPParserRegistration:
    """HWPParser의 확장자 매칭을 검증합니다."""

    def test_확장자(self):
        """지원 확장자가 .hwp인지 확인합니다."""
        from slm_factory.parsers.hwp import HWPParser

        assert HWPParser.extensions == [".hwp"]

    def test_can_parse_hwp(self, tmp_path: Path):
        """.hwp 파일에 대해 can_parse가 True를 반환합니다."""
        from slm_factory.parsers.hwp import HWPParser

        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"dummy")
        parser = HWPParser()
        assert parser.can_parse(hwp_file) is True

    def test_cannot_parse_txt(self, tmp_path: Path):
        """.txt 파일에 대해 can_parse가 False를 반환합니다."""
        from slm_factory.parsers.hwp import HWPParser

        txt = tmp_path / "file.txt"
        txt.write_text("text")
        parser = HWPParser()
        assert parser.can_parse(txt) is False


# ---------------------------------------------------------------------------
# 바이너리 레코드 파싱 유닛 테스트
# ---------------------------------------------------------------------------


class TestDecodeParaText:
    """_decode_para_text 함수를 직접 검증합니다."""

    def test_순수_텍스트_디코딩(self):
        """제어 코드 없는 순수 텍스트를 올바르게 디코딩합니다."""
        from slm_factory.parsers.hwp import _decode_para_text

        text = "한글 테스트"
        payload = text.encode("utf-16-le")
        assert _decode_para_text(payload) == text

    def test_제어_코드_건너뜀(self):
        """인라인 제어 코드(0~31)를 건너뛰고 텍스트만 추출합니다."""
        from slm_factory.parsers.hwp import _decode_para_text

        payload = b""
        payload += "가".encode("utf-16-le")
        payload += struct.pack("<H", 5)
        payload += "나".encode("utf-16-le")
        result = _decode_para_text(payload)
        assert "가" in result
        assert "나" in result

    def test_줄바꿈_처리(self):
        """CR(13), LF(10) 제어 코드를 줄바꿈으로 변환합니다."""
        from slm_factory.parsers.hwp import _decode_para_text

        payload = b""
        payload += "가".encode("utf-16-le")
        payload += struct.pack("<H", 13)
        payload += "나".encode("utf-16-le")
        result = _decode_para_text(payload)
        assert "\n" in result


class TestExtractTextFromBodytext:
    """_extract_text_from_bodytext 레코드 파싱을 검증합니다."""

    def test_단일_레코드_파싱(self):
        """단일 HWPTAG_PARA_TEXT 레코드에서 텍스트를 추출합니다."""
        from slm_factory.parsers.hwp import _extract_text_from_bodytext

        data = _build_hwptag_para_text("테스트 단락")
        result = _extract_text_from_bodytext(data)
        assert len(result) == 1
        assert "테스트 단락" in result[0]

    def test_여러_레코드_파싱(self):
        """여러 HWPTAG_PARA_TEXT 레코드에서 텍스트를 추출합니다."""
        from slm_factory.parsers.hwp import _extract_text_from_bodytext

        data = _build_bodytext_data("단락1", "단락2", "단락3")
        result = _extract_text_from_bodytext(data)
        assert len(result) == 3

    def test_빈_데이터(self):
        """빈 데이터에서 빈 리스트를 반환합니다."""
        from slm_factory.parsers.hwp import _extract_text_from_bodytext

        result = _extract_text_from_bodytext(b"")
        assert result == []
