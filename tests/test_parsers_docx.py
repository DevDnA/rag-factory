"""DOCX 파서(parsers/docx.py)의 단위 테스트입니다."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.modules["docx"] = MagicMock()

from slm_factory.parsers.docx import DOCXParser


class TestDOCXParser:
    """DOCXParser 클래스의 테스트입니다."""

    def test_can_parse_docx_True(self):
        """.docx 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = DOCXParser()
        assert parser.can_parse(Path("document.docx")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = DOCXParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_ImportError_when_docx_not_installed(self, tmp_path):
        """python-docx가 설치되지 않았을 때 ImportError를 발생시키는지 확인합니다."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        parser = DOCXParser()

        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ImportError) as exc_info:
                parser.parse(docx_file)

            assert "python-docx가 설치되지 않았습니다" in str(exc_info.value)
            assert "uv sync --extra docx" in str(exc_info.value)

    def test_parse_FileNotFoundError_when_file_missing(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = DOCXParser()

        mock_doc = MagicMock()
        with patch("docx.Document", return_value=mock_doc):
            with pytest.raises(FileNotFoundError):
                parser.parse(tmp_path / "nonexistent.docx")

    def test_parse_기본_DOCX(self, tmp_path):
        """기본 DOCX 파일을 파싱하여 ParsedDocument를 반환하는지 확인합니다."""
        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            Mock(text="첫 번째 단락입니다.", style=Mock(name="Normal")),
            Mock(text="두 번째 단락입니다.", style=Mock(name="Normal")),
        ]
        mock_doc.tables = []
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert doc.doc_id == "document.docx"
            assert isinstance(doc.content, str)
            assert "첫 번째 단락" in doc.content
            assert "두 번째 단락" in doc.content

    def test_parse_제목_추출_from_properties(self, tmp_path):
        """DOCX의 core_properties에서 제목을 추출하는지 확인합니다."""
        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.paragraphs = [Mock(text="내용", style=Mock(name="Normal"))]
        mock_doc.tables = []
        mock_doc.core_properties.title = "문서 제목"
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert doc.title == "문서 제목"

    def test_parse_제목_폴백_to_filename(self, tmp_path):
        """core_properties.title이 없으면 파일명에서 제목을 추출하는지 확인합니다."""
        docx_file = tmp_path / "my_document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.paragraphs = [Mock(text="내용", style=Mock(name="Normal"))]
        mock_doc.tables = []
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert doc.title == "my_document"

    def test_parse_헤딩_스타일_처리(self, tmp_path):
        """Heading 스타일을 마크다운 헤딩으로 변환하는지 확인합니다."""
        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        style1 = Mock()
        style1.name = "Heading 1"

        style2 = Mock()
        style2.name = "Heading 2"

        style3 = Mock()
        style3.name = "Heading 3"

        style_normal = Mock()
        style_normal.name = "Normal"

        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            Mock(text="제목 1", style=style1),
            Mock(text="제목 2", style=style2),
            Mock(text="제목 3", style=style3),
            Mock(text="일반 텍스트", style=style_normal),
        ]
        mock_doc.tables = []
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert "# 제목 1" in doc.content
            assert "## 제목 2" in doc.content
            assert "### 제목 3" in doc.content
            assert "일반 텍스트" in doc.content

    def test_parse_표_추출(self, tmp_path):
        """DOCX의 표를 마크다운 표로 변환하여 tables에 저장하는지 확인합니다."""
        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_row1 = Mock()
        mock_row1.cells = [Mock(text="이름"), Mock(text="값")]

        mock_row2 = Mock()
        mock_row2.cells = [Mock(text="A"), Mock(text="1")]

        mock_row3 = Mock()
        mock_row3.cells = [Mock(text="B"), Mock(text="2")]

        mock_table = Mock()
        mock_table.rows = [mock_row1, mock_row2, mock_row3]

        mock_doc = MagicMock()
        mock_doc.paragraphs = [Mock(text="내용", style=Mock(name="Normal"))]
        mock_doc.tables = [mock_table]
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert len(doc.tables) == 1
            assert "| 이름 | 값 |" in doc.tables[0]
            assert "| --- | --- |" in doc.tables[0]
            assert "| A | 1 |" in doc.tables[0]
            assert "| B | 2 |" in doc.tables[0]

    def test_parse_메타데이터_추출(self, tmp_path):
        """DOCX의 core_properties에서 메타데이터를 추출하는지 확인합니다."""
        from datetime import datetime

        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.paragraphs = [Mock(text="내용", style=Mock(name="Normal"))]
        mock_doc.tables = []
        mock_doc.core_properties.title = "제목"
        mock_doc.core_properties.author = "홍길동"
        mock_doc.core_properties.created = datetime(2024, 1, 15, 10, 30, 0)

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert doc.metadata["author"] == "홍길동"
            assert doc.metadata["date"] == "2024-01-15"

    def test_parse_빈_단락_제외(self, tmp_path):
        """빈 단락은 콘텐츠에서 제외되는지 확인합니다."""
        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            Mock(text="첫 번째 단락", style=Mock(name="Normal")),
            Mock(text="   ", style=Mock(name="Normal")),
            Mock(text="", style=Mock(name="Normal")),
            Mock(text="두 번째 단락", style=Mock(name="Normal")),
        ]
        mock_doc.tables = []
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert "첫 번째 단락" in doc.content
            assert "두 번째 단락" in doc.content
            lines = [line for line in doc.content.split("\n\n") if line.strip()]
            assert len(lines) == 2

    def test_parse_파일명에서_날짜_추출(self, tmp_path):
        """파일명에서 YYMMDD 날짜를 추출하여 메타데이터에 저장하는지 확인합니다."""
        docx_file = tmp_path / "report_240115.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.paragraphs = [Mock(text="내용", style=Mock(name="Normal"))]
        mock_doc.tables = []
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert doc.metadata.get("date") == "2024-01-15"

    def test_parse_표_헤더만_있는_경우(self, tmp_path):
        """헤더만 있는 표를 올바르게 처리하는지 확인합니다."""
        docx_file = tmp_path / "document.docx"
        docx_file.write_text("dummy", encoding="utf-8")

        mock_row1 = Mock()
        mock_row1.cells = [Mock(text="열1"), Mock(text="열2")]

        mock_table = Mock()
        mock_table.rows = [mock_row1]

        mock_doc = MagicMock()
        mock_doc.paragraphs = [Mock(text="내용", style=Mock(name="Normal"))]
        mock_doc.tables = [mock_table]
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.created = None

        parser = DOCXParser()

        with patch("docx.Document", return_value=mock_doc):
            doc = parser.parse(docx_file)

            assert len(doc.tables) == 1
            assert "| 열1 | 열2 |" in doc.tables[0]
            assert "| --- | --- |" in doc.tables[0]
