"""DOC 파서(parsers/doc.py)의 단위 테스트입니다."""

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.modules["olefile"] = MagicMock()

from slm_factory.parsers.doc import DOCParser


def _build_minimal_word_doc(text: str) -> bytes:
    """최소한의 유효한 WordDocument 스트림 바이트를 구성합니다.

    FIB magic(0xA5EC), flags, fcClx, lcbClx 필드를 포함하며
    텍스트 데이터를 뒤에 붙입니다. 테스트용 단순 구조입니다.
    """
    # _FIB_LCBCLX_OFFSET = 0x01A6 (422) 까지 패딩이 필요합니다.
    header_size = 0x01A6 + 4  # 426 bytes
    buf = bytearray(header_size + 512)

    # magic
    struct.pack_into("<H", buf, 0, 0xA5EC)

    # flags (bit 9 = 0 → "0Table")
    struct.pack_into("<H", buf, 0x000A, 0x0000)

    # CLX: type 0x02 + pcdt_size(4) + plc_data
    # plc_data: 2 CPs (각 4바이트) + 1 PCD (8바이트) = 16바이트
    # CP[0]=0, CP[1]=char_count, PCD: unused(2) + fc_raw(4) + prm(2)
    # 텍스트는 UTF-16LE로 word_doc 스트림 뒤에 붙임
    text_utf16 = text.encode("utf-16-le")
    char_count = len(text_utf16) // 2

    # 텍스트 위치: header_size 이후
    text_start = header_size
    # fCompressed=0 → fc = byte offset, 유니코드
    fc_raw = text_start  # bit 30 clear = unicode

    plc = bytearray(16)
    struct.pack_into("<I", plc, 0, 0)            # CP[0] = 0
    struct.pack_into("<I", plc, 4, char_count)   # CP[1] = char_count
    struct.pack_into("<H", plc, 8, 0)            # PCD unused
    struct.pack_into("<I", plc, 10, fc_raw)      # PCD fc_raw
    struct.pack_into("<H", plc, 14, 0)           # PCD prm

    clx = bytearray()
    clx.append(0x02)                             # Pcdt type
    clx += struct.pack("<I", len(plc))           # pcdt_size
    clx += plc

    clx_start = len(buf) - 256
    buf[clx_start : clx_start + len(clx)] = clx

    # fcClx, lcbClx
    struct.pack_into("<I", buf, 0x01A2, clx_start)
    struct.pack_into("<I", buf, 0x01A6, len(clx))

    # 텍스트 데이터 붙이기
    final = bytes(buf) + text_utf16
    return final


def _build_ppt_text_chars_atom(text: str) -> bytes:
    """TextCharsAtom 레코드를 포함하는 최소 PPT 스트림을 만듭니다."""
    _RT_TEXT_CHARS_ATOM = 0x0FA0
    payload = text.encode("utf-16-le")
    rec_ver_instance = 0x0000
    header = struct.pack("<HHI", rec_ver_instance, _RT_TEXT_CHARS_ATOM, len(payload))
    return header + payload


class TestDOCParser:
    """DOCParser 클래스의 테스트입니다."""

    def test_can_parse_doc_True(self):
        """.doc 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = DOCParser()
        assert parser.can_parse(Path("document.doc")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = DOCParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_FileNotFoundError_when_file_missing(self, tmp_path):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = DOCParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.doc")

    def test_parse_RuntimeError_when_olefile_not_installed(self, tmp_path):
        """olefile이 설치되지 않았을 때 RuntimeError를 발생시키는지 확인합니다."""
        doc_file = tmp_path / "test.doc"
        doc_file.write_bytes(b"dummy")

        parser = DOCParser()

        with patch("slm_factory.parsers.doc.HAS_OLEFILE", False):
            with pytest.raises(RuntimeError) as exc_info:
                parser.parse(doc_file)

            assert "olefile" in str(exc_info.value)
            assert "uv sync --extra hwp" in str(exc_info.value)

    def test_parse_RuntimeError_when_invalid_ole(self, tmp_path):
        """유효한 OLE 파일이 아닐 때 RuntimeError를 발생시키는지 확인합니다."""
        doc_file = tmp_path / "test.doc"
        doc_file.write_bytes(b"not an ole file")

        parser = DOCParser()

        with patch("slm_factory.parsers.doc.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.doc.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = False
                with pytest.raises(RuntimeError) as exc_info:
                    parser.parse(doc_file)

                assert "OLE2" in str(exc_info.value) or "DOC" in str(exc_info.value)

    def test_parse_기본_DOC(self, tmp_path):
        """기본 DOC 파일을 파싱하여 한국어 텍스트를 추출하는지 확인합니다."""
        doc_file = tmp_path / "document.doc"
        doc_file.write_bytes(b"dummy")

        korean_text = "첫 번째 단락입니다"
        word_doc_bytes = _build_minimal_word_doc(korean_text)

        mock_stream = MagicMock()
        mock_stream.read.return_value = word_doc_bytes

        mock_table_stream = MagicMock()
        # 테이블 스트림은 word_doc_bytes 자체에 CLX를 포함하므로 word_doc_bytes 반환
        mock_table_stream.read.return_value = word_doc_bytes

        mock_ole = MagicMock()
        mock_ole.openstream.side_effect = lambda name: (
            mock_stream if name == "WordDocument" else mock_table_stream
        )
        mock_ole.exists.return_value = True
        mock_ole.get_metadata.side_effect = Exception("no metadata")

        parser = DOCParser()

        with patch("slm_factory.parsers.doc.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.doc.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = True
                mock_ole_mod.OleFileIO.return_value = mock_ole

                doc = parser.parse(doc_file)

                assert doc.doc_id == "document.doc"
                assert isinstance(doc.content, str)
                assert isinstance(doc.title, str)
                assert doc.title == "document"

    def test_parse_빈_DOC(self, tmp_path):
        """텍스트가 없는 DOC 파일을 파싱하면 빈 content가 반환되는지 확인합니다."""
        doc_file = tmp_path / "empty.doc"
        doc_file.write_bytes(b"dummy")

        # FIB magic 없는 짧은 바이트 → fallback → 빈 결과
        word_doc_bytes = b"\x00" * 8

        mock_stream = MagicMock()
        mock_stream.read.return_value = word_doc_bytes

        mock_ole = MagicMock()
        mock_ole.openstream.return_value = mock_stream
        mock_ole.exists.return_value = False
        mock_ole.get_metadata.side_effect = Exception("no metadata")

        parser = DOCParser()

        with patch("slm_factory.parsers.doc.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.doc.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = True
                mock_ole_mod.OleFileIO.return_value = mock_ole

                doc = parser.parse(doc_file)

                assert doc.doc_id == "empty.doc"
                assert doc.content == ""

    def test_parse_메타데이터_추출(self, tmp_path):
        """DOC 파일의 OLE 메타데이터에서 author와 title을 추출하는지 확인합니다."""
        from datetime import datetime

        doc_file = tmp_path / "report.doc"
        doc_file.write_bytes(b"dummy")

        # fallback 텍스트 추출: UTF-16LE 텍스트를 포함하는 워드 문서
        content_text = "테스트 내용입니다."
        word_doc_bytes = content_text.encode("utf-16-le")

        mock_stream = MagicMock()
        mock_stream.read.return_value = word_doc_bytes

        mock_meta = MagicMock()
        mock_meta.author = "홍길동"
        mock_meta.title = "테스트 제목"
        mock_meta.create_time = datetime(2024, 3, 15)

        mock_ole = MagicMock()
        mock_ole.openstream.return_value = mock_stream
        mock_ole.exists.return_value = False
        mock_ole.get_metadata.return_value = mock_meta

        parser = DOCParser()

        with patch("slm_factory.parsers.doc.HAS_OLEFILE", True):
            with patch("slm_factory.parsers.doc.olefile") as mock_ole_mod:
                mock_ole_mod.isOleFile.return_value = True
                mock_ole_mod.OleFileIO.return_value = mock_ole

                doc = parser.parse(doc_file)

                assert doc.metadata.get("author") == "홍길동"
                assert doc.title == "테스트 제목"
                assert doc.metadata.get("date") == "2024-03-15"
