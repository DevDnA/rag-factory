"""XLS 파서(parsers/xls.py)의 단위 테스트입니다."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.modules["xlrd"] = MagicMock()

from slm_factory.parsers.xls import XLSParser


class TestXLSParser:
    """XLSParser 클래스의 테스트입니다."""

    def test_can_parse_xls_True(self):
        """.xls 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = XLSParser()
        assert parser.can_parse(Path("legacy.xls")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = XLSParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_ImportError_when_xlrd_not_installed(self, tmp_path):
        """xlrd가 설치되지 않았을 때 ImportError를 발생시키는지 확인합니다."""
        xls_file = tmp_path / "test.xls"
        xls_file.write_text("dummy", encoding="utf-8")

        parser = XLSParser()

        with patch.dict("sys.modules", {"xlrd": None}):
            with pytest.raises(ImportError) as exc_info:
                parser.parse(xls_file)

            assert "xlrd가 설치되지 않았습니다" in str(exc_info.value)
            assert "uv sync --extra xls" in str(exc_info.value)

    def test_parse_FileNotFoundError_when_file_missing(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = XLSParser()

        mock_wb = MagicMock()
        with patch("xlrd.open_workbook", return_value=mock_wb):
            with pytest.raises(FileNotFoundError):
                parser.parse(tmp_path / "nonexistent.xls")

    def test_parse_기본_XLS(self, tmp_path):
        """기본 XLS 파일을 파싱하여 ParsedDocument를 반환하는지 확인합니다."""
        xls_file = tmp_path / "legacy.xls"
        xls_file.write_text("dummy", encoding="utf-8")

        # 셀 설정: xlrd cell 객체는 .value 속성을 가짐
        def make_cell(value):
            c = Mock()
            c.value = value
            return c

        mock_sheet = Mock()
        mock_sheet.name = "Sheet1"
        mock_sheet.nrows = 3
        mock_sheet.ncols = 2
        mock_sheet.cell = Mock(
            side_effect=lambda r, c: [
                [make_cell("이름"), make_cell("나이")],
                [make_cell("김철수"), make_cell(30.0)],
                [make_cell("이영희"), make_cell(25.0)],
            ][r][c]
        )

        mock_wb = MagicMock()
        mock_wb.nsheets = 1
        mock_wb.sheet_by_index = Mock(return_value=mock_sheet)

        parser = XLSParser()

        with patch("xlrd.open_workbook", return_value=mock_wb):
            doc = parser.parse(xls_file)

            assert doc.doc_id == "legacy.xls"
            assert isinstance(doc.content, str)
            assert "김철수" in doc.content
            assert "이영희" in doc.content
            # float 정수는 int로 변환되어야 함 (30.0 -> "30")
            assert "30" in doc.content
            assert len(doc.tables) == 1
            assert "| 이름 | 나이 |" in doc.tables[0]

    def test_parse_여러_시트(self, tmp_path):
        """여러 시트가 있는 XLS 파일에서 각 시트를 마크다운 표로 변환하는지 확인합니다."""
        xls_file = tmp_path / "multi_sheet.xls"
        xls_file.write_text("dummy", encoding="utf-8")

        def make_cell(value):
            c = Mock()
            c.value = value
            return c

        # 시트 1: 재고
        mock_sheet1 = Mock()
        mock_sheet1.name = "재고"
        mock_sheet1.nrows = 2
        mock_sheet1.ncols = 2
        sheet1_data = [
            [make_cell("제품"), make_cell("수량")],
            [make_cell("노트북"), make_cell(5.0)],
        ]
        mock_sheet1.cell = Mock(side_effect=lambda r, c: sheet1_data[r][c])

        # 시트 2: 매출
        mock_sheet2 = Mock()
        mock_sheet2.name = "매출"
        mock_sheet2.nrows = 2
        mock_sheet2.ncols = 2
        sheet2_data = [
            [make_cell("월"), make_cell("금액")],
            [make_cell("1월"), make_cell(1000000.0)],
        ]
        mock_sheet2.cell = Mock(side_effect=lambda r, c: sheet2_data[r][c])

        mock_wb = MagicMock()
        mock_wb.nsheets = 2
        mock_wb.sheet_by_index = Mock(side_effect=lambda i: [mock_sheet1, mock_sheet2][i])

        parser = XLSParser()

        with patch("xlrd.open_workbook", return_value=mock_wb):
            doc = parser.parse(xls_file)

            assert len(doc.tables) == 2
            assert "## 재고" in doc.content
            assert "## 매출" in doc.content
            assert "노트북" in doc.content
            assert "1000000" in doc.content

    def test_parse_빈_XLS(self, tmp_path):
        """행이 없는 XLS 시트를 파싱하면 빈 content를 반환하는지 확인합니다."""
        xls_file = tmp_path / "empty.xls"
        xls_file.write_text("dummy", encoding="utf-8")

        mock_sheet = Mock()
        mock_sheet.name = "Sheet1"
        mock_sheet.nrows = 0
        mock_sheet.ncols = 0

        mock_wb = MagicMock()
        mock_wb.nsheets = 1
        mock_wb.sheet_by_index = Mock(return_value=mock_sheet)

        parser = XLSParser()

        with patch("xlrd.open_workbook", return_value=mock_wb):
            doc = parser.parse(xls_file)

            assert doc.doc_id == "empty.xls"
            assert doc.content == ""
            assert doc.tables == []
            assert doc.metadata["sheet_count"] == 1
