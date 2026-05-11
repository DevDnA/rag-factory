"""텍스트 파서(parsers/text.py)의 단위 테스트입니다."""

from pathlib import Path

import pytest

from slm_factory.parsers.base import detect_encoding
from slm_factory.parsers.text import TextParser


# ---------------------------------------------------------------------------
# detect_encoding (shared utility in parsers/base.py)
# ---------------------------------------------------------------------------


class TestDetectEncoding:
    """detect_encoding 함수의 테스트입니다."""

    def test_utf8_콘텐츠(self):
        """유효한 UTF-8 바이트 시퀀스에 대해 'utf-8'을 반환하는지 확인합니다."""
        content = "한글 텍스트입니다.".encode("utf-8")
        assert detect_encoding(content) == "utf-8"

    def test_비_utf8_콘텐츠(self):
        """UTF-8로 디코딩할 수 없는 바이트에 대해 적절한 인코딩을 반환하는지 확인합니다."""
        # 유효하지 않은 UTF-8 시퀀스 — charset-normalizer가 최적 인코딩을 추론
        content = b"\xff\xfe\xfd\x80\x81"
        result = detect_encoding(content)
        # charset-normalizer는 latin-1 대신 다른 인코딩을 반환할 수 있음
        assert isinstance(result, str)
        assert len(result) > 0

    def test_빈_콘텐츠_utf8_폴백(self):
        """빈 바이트에 대해 'utf-8'을 반환하는지 확인합니다."""
        assert detect_encoding(b"") == "utf-8"


# ---------------------------------------------------------------------------
# TextParser
# ---------------------------------------------------------------------------


class TestTextParser:
    """TextParser 클래스의 테스트입니다."""

    def test_can_parse_txt_True(self):
        """.txt 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = TextParser()
        assert parser.can_parse(Path("file.txt")) is True

    def test_can_parse_md_True(self):
        """.md 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = TextParser()
        assert parser.can_parse(Path("file.md")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = TextParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_txt_파일(self, tmp_text_file):
        """.txt 파일을 파싱하면 제목이 파일명 stem인지 확인합니다."""
        parser = TextParser()
        doc = parser.parse(tmp_text_file)

        assert doc.doc_id == "sample.txt"
        assert doc.title == "sample"  # .txt는 파일명 stem을 제목으로 사용
        assert "테스트용 텍스트 파일" in doc.content
        assert doc.tables == []

    def test_parse_md_파일(self, tmp_md_file):
        """.md 파일을 파싱하면 첫 번째 # 제목이 추출되는지 확인합니다."""
        parser = TextParser()
        doc = parser.parse(tmp_md_file)

        assert doc.doc_id == "readme.md"
        assert doc.title == "테스트 마크다운"  # # 제목에서 추출
        assert "본문 내용" in doc.content

    def test_parse_존재하지_않는_파일(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = TextParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.txt")

    def test_parse_날짜_포함_파일명(self, tmp_path):
        """YYMMDD 날짜가 포함된 파일명에서 metadata에 date가 설정되는지 확인합니다."""
        dated_file = tmp_path / "report_240115_final.txt"
        dated_file.write_text("보고서 내용입니다.", encoding="utf-8")

        parser = TextParser()
        doc = parser.parse(dated_file)

        assert "date" in doc.metadata
        assert doc.metadata["date"] == "2024-01-15"
