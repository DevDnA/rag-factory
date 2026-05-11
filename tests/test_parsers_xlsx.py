"""XLSX 파서(parsers/xlsx.py)의 단위 테스트입니다."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.modules["openpyxl"] = MagicMock()

from slm_factory.parsers.xlsx import XLSXParser


class TestXLSXParser:
    """XLSXParser 클래스의 테스트입니다."""

    def test_can_parse_xlsx_True(self):
        """.xlsx 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = XLSXParser()
        assert parser.can_parse(Path("spreadsheet.xlsx")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = XLSXParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_ImportError_when_openpyxl_not_installed(self, tmp_path):
        """openpyxl이 설치되지 않았을 때 ImportError를 발생시키는지 확인합니다."""
        xlsx_file = tmp_path / "test.xlsx"
        xlsx_file.write_text("dummy", encoding="utf-8")

        parser = XLSXParser()

        with patch.dict("sys.modules", {"openpyxl": None}):
            with pytest.raises(ImportError) as exc_info:
                parser.parse(xlsx_file)

            assert "openpyxl이 설치되지 않았습니다" in str(exc_info.value)
            assert "uv sync --extra xlsx" in str(exc_info.value)

    def test_parse_FileNotFoundError_when_file_missing(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = XLSXParser()

        mock_wb = MagicMock()
        with patch("openpyxl.load_workbook", return_value=mock_wb):
            with pytest.raises(FileNotFoundError):
                parser.parse(tmp_path / "nonexistent.xlsx")

    def test_parse_기본_XLSX(self, tmp_path):
        """기본 XLSX 파일을 파싱하여 ParsedDocument를 반환하는지 확인합니다."""
        xlsx_file = tmp_path / "spreadsheet.xlsx"
        xlsx_file.write_text("dummy", encoding="utf-8")

        # 셀 행 설정
        def make_cell(value):
            c = Mock()
            c.value = value
            return c

        row1 = [make_cell("이름"), make_cell("점수")]
        row2 = [make_cell("김철수"), make_cell("95")]
        row3 = [make_cell("이영희"), make_cell("87")]

        mock_sheet = Mock()
        mock_sheet.title = "Sheet1"
        mock_sheet.iter_rows = Mock(return_value=iter([row1, row2, row3]))

        mock_wb = MagicMock()
        mock_wb.worksheets = [mock_sheet]
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.properties.creator = None
        mock_wb.properties.title = None
        mock_wb.properties.created = None
        mock_wb.close = Mock()

        parser = XLSXParser()

        with patch("openpyxl.load_workbook", return_value=mock_wb):
            doc = parser.parse(xlsx_file)

            assert doc.doc_id == "spreadsheet.xlsx"
            assert isinstance(doc.content, str)
            assert "김철수" in doc.content
            assert "이영희" in doc.content
            assert len(doc.tables) == 1
            assert "| 이름 | 점수 |" in doc.tables[0]
            assert "| 김철수 | 95 |" in doc.tables[0]

    def test_parse_여러_시트(self, tmp_path):
        """여러 시트가 있는 XLSX 파일에서 각 시트를 마크다운 표로 변환하는지 확인합니다."""
        xlsx_file = tmp_path / "multi_sheet.xlsx"
        xlsx_file.write_text("dummy", encoding="utf-8")

        def make_cell(value):
            c = Mock()
            c.value = value
            return c

        # 시트 1
        sheet1_rows = [
            [make_cell("제품"), make_cell("수량")],
            [make_cell("사과"), make_cell("10")],
        ]
        mock_sheet1 = Mock()
        mock_sheet1.title = "재고"
        mock_sheet1.iter_rows = Mock(return_value=iter(sheet1_rows))

        # 시트 2
        sheet2_rows = [
            [make_cell("날짜"), make_cell("매출")],
            [make_cell("2024-01"), make_cell("500000")],
        ]
        mock_sheet2 = Mock()
        mock_sheet2.title = "매출"
        mock_sheet2.iter_rows = Mock(return_value=iter(sheet2_rows))

        mock_wb = MagicMock()
        mock_wb.worksheets = [mock_sheet1, mock_sheet2]
        mock_wb.sheetnames = ["재고", "매출"]
        mock_wb.properties.creator = None
        mock_wb.properties.title = None
        mock_wb.properties.created = None
        mock_wb.close = Mock()

        parser = XLSXParser()

        with patch("openpyxl.load_workbook", return_value=mock_wb):
            doc = parser.parse(xlsx_file)

            assert len(doc.tables) == 2
            assert "## 재고" in doc.content
            assert "## 매출" in doc.content
            assert "사과" in doc.content
            assert "500000" in doc.content

    def test_parse_빈_XLSX(self, tmp_path):
        """빈 시트가 있는 XLSX 파일을 파싱하면 빈 content를 반환하는지 확인합니다."""
        xlsx_file = tmp_path / "empty.xlsx"
        xlsx_file.write_text("dummy", encoding="utf-8")

        # 모든 셀이 None인 빈 행
        def make_empty_cell():
            c = Mock()
            c.value = None
            return c

        empty_row = [make_empty_cell(), make_empty_cell()]
        mock_sheet = Mock()
        mock_sheet.title = "Sheet1"
        mock_sheet.iter_rows = Mock(return_value=iter([empty_row]))

        mock_wb = MagicMock()
        mock_wb.worksheets = [mock_sheet]
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.properties.creator = None
        mock_wb.properties.title = None
        mock_wb.properties.created = None
        mock_wb.close = Mock()

        parser = XLSXParser()

        with patch("openpyxl.load_workbook", return_value=mock_wb):
            doc = parser.parse(xlsx_file)

            assert doc.doc_id == "empty.xlsx"
            assert doc.content == ""
            assert doc.tables == []

    def test_parse_메타데이터_추출(self, tmp_path):
        """XLSX의 properties에서 메타데이터를 추출하는지 확인합니다."""
        from datetime import datetime

        xlsx_file = tmp_path / "spreadsheet.xlsx"
        xlsx_file.write_text("dummy", encoding="utf-8")

        def make_cell(value):
            c = Mock()
            c.value = value
            return c

        row1 = [make_cell("항목"), make_cell("값")]
        row2 = [make_cell("테스트"), make_cell("123")]

        mock_sheet = Mock()
        mock_sheet.title = "Sheet1"
        mock_sheet.iter_rows = Mock(return_value=iter([row1, row2]))

        mock_wb = MagicMock()
        mock_wb.worksheets = [mock_sheet]
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.properties.creator = "홍길동"
        mock_wb.properties.title = "분기 보고서"
        mock_wb.properties.created = datetime(2024, 6, 1, 0, 0, 0)
        mock_wb.close = Mock()

        parser = XLSXParser()

        with patch("openpyxl.load_workbook", return_value=mock_wb):
            doc = parser.parse(xlsx_file)

            assert doc.title == "분기 보고서"
            assert doc.metadata["author"] == "홍길동"
            assert doc.metadata["date"] == "2024-06-01"
            assert doc.metadata["sheet_count"] == 1
