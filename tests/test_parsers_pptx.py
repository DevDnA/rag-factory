"""PPTX 파서(parsers/pptx.py)의 단위 테스트입니다."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.modules["pptx"] = MagicMock()
sys.modules["pptx.table"] = MagicMock()

from slm_factory.parsers.pptx import PPTXParser


class TestPPTXParser:
    """PPTXParser 클래스의 테스트입니다."""

    def test_can_parse_pptx_True(self):
        """.pptx 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = PPTXParser()
        assert parser.can_parse(Path("presentation.pptx")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = PPTXParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_ImportError_when_pptx_not_installed(self, tmp_path):
        """python-pptx가 설치되지 않았을 때 ImportError를 발생시키는지 확인합니다."""
        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_text("dummy", encoding="utf-8")

        parser = PPTXParser()

        with patch.dict("sys.modules", {"pptx": None, "pptx.table": None}):
            with pytest.raises(ImportError) as exc_info:
                parser.parse(pptx_file)

            assert "python-pptx가 설치되지 않았습니다" in str(exc_info.value)
            assert "uv sync --extra pptx" in str(exc_info.value)

    def test_parse_FileNotFoundError_when_file_missing(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = PPTXParser()

        mock_prs = MagicMock()
        with patch("pptx.Presentation", return_value=mock_prs):
            with pytest.raises(FileNotFoundError):
                parser.parse(tmp_path / "nonexistent.pptx")

    def test_parse_기본_PPTX(self, tmp_path):
        """기본 PPTX 파일을 파싱하여 ParsedDocument를 반환하는지 확인합니다."""
        pptx_file = tmp_path / "presentation.pptx"
        pptx_file.write_text("dummy", encoding="utf-8")

        # 텍스트 프레임이 있는 shape 설정
        mock_para1 = Mock()
        mock_para1.text = "슬라이드 제목"

        mock_para2 = Mock()
        mock_para2.text = "슬라이드 내용입니다."

        mock_text_frame = Mock()
        mock_text_frame.paragraphs = [mock_para1, mock_para2]

        mock_shape = Mock()
        mock_shape.has_table = False
        mock_shape.has_text_frame = True
        mock_shape.text_frame = mock_text_frame

        mock_slide = Mock()
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_prs.core_properties.title = None
        mock_prs.core_properties.author = None
        mock_prs.core_properties.created = None

        parser = PPTXParser()

        with patch("pptx.Presentation", return_value=mock_prs):
            doc = parser.parse(pptx_file)

            assert doc.doc_id == "presentation.pptx"
            assert isinstance(doc.content, str)
            assert "슬라이드 제목" in doc.content
            assert "슬라이드 내용입니다." in doc.content

    def test_parse_표_추출(self, tmp_path):
        """PPTX 슬라이드의 표를 마크다운 표로 변환하여 tables에 저장하는지 확인합니다."""
        pptx_file = tmp_path / "presentation.pptx"
        pptx_file.write_text("dummy", encoding="utf-8")

        # 표 셀 설정
        mock_cell_name = Mock()
        mock_cell_name.text = "이름"
        mock_cell_value = Mock()
        mock_cell_value.text = "값"

        mock_cell_a = Mock()
        mock_cell_a.text = "항목A"
        mock_cell_1 = Mock()
        mock_cell_1.text = "100"

        mock_cell_b = Mock()
        mock_cell_b.text = "항목B"
        mock_cell_2 = Mock()
        mock_cell_2.text = "200"

        mock_row1 = Mock()
        mock_row1.cells = [mock_cell_name, mock_cell_value]

        mock_row2 = Mock()
        mock_row2.cells = [mock_cell_a, mock_cell_1]

        mock_row3 = Mock()
        mock_row3.cells = [mock_cell_b, mock_cell_2]

        mock_table = Mock()
        mock_table.rows = [mock_row1, mock_row2, mock_row3]

        mock_shape = Mock()
        mock_shape.has_table = True
        mock_shape.table = mock_table

        mock_slide = Mock()
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_prs.core_properties.title = None
        mock_prs.core_properties.author = None
        mock_prs.core_properties.created = None

        parser = PPTXParser()

        with patch("pptx.Presentation", return_value=mock_prs):
            doc = parser.parse(pptx_file)

            assert len(doc.tables) == 1
            assert "| 이름 | 값 |" in doc.tables[0]
            assert "| --- | --- |" in doc.tables[0]
            assert "| 항목A | 100 |" in doc.tables[0]
            assert "| 항목B | 200 |" in doc.tables[0]

    def test_parse_노트_추출(self, tmp_path):
        """슬라이드 발표자 노트를 추출하는지 확인합니다."""
        pptx_file = tmp_path / "presentation.pptx"
        pptx_file.write_text("dummy", encoding="utf-8")

        mock_para = Mock()
        mock_para.text = "슬라이드 본문"

        mock_text_frame = Mock()
        mock_text_frame.paragraphs = [mock_para]

        mock_shape = Mock()
        mock_shape.has_table = False
        mock_shape.has_text_frame = True
        mock_shape.text_frame = mock_text_frame

        mock_notes_text_frame = Mock()
        mock_notes_text_frame.text = "이것은 발표자 노트입니다."

        mock_notes_slide = Mock()
        mock_notes_slide.notes_text_frame = mock_notes_text_frame

        mock_slide = Mock()
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = True
        mock_slide.notes_slide = mock_notes_slide

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_prs.core_properties.title = None
        mock_prs.core_properties.author = None
        mock_prs.core_properties.created = None

        parser = PPTXParser()

        with patch("pptx.Presentation", return_value=mock_prs):
            doc = parser.parse(pptx_file)

            assert "노트" in doc.content
            assert "이것은 발표자 노트입니다." in doc.content

    def test_parse_빈_PPTX(self, tmp_path):
        """슬라이드가 없는 PPTX 파일을 파싱하면 빈 content를 반환하는지 확인합니다."""
        pptx_file = tmp_path / "empty.pptx"
        pptx_file.write_text("dummy", encoding="utf-8")

        mock_prs = MagicMock()
        mock_prs.slides = []
        mock_prs.core_properties.title = None
        mock_prs.core_properties.author = None
        mock_prs.core_properties.created = None

        parser = PPTXParser()

        with patch("pptx.Presentation", return_value=mock_prs):
            doc = parser.parse(pptx_file)

            assert doc.doc_id == "empty.pptx"
            assert doc.content == ""
            assert doc.tables == []
            assert doc.metadata["slide_count"] == 0

    def test_parse_메타데이터_추출(self, tmp_path):
        """PPTX의 core_properties에서 메타데이터를 추출하는지 확인합니다."""
        from datetime import datetime

        pptx_file = tmp_path / "presentation.pptx"
        pptx_file.write_text("dummy", encoding="utf-8")

        mock_para = Mock()
        mock_para.text = "내용"

        mock_text_frame = Mock()
        mock_text_frame.paragraphs = [mock_para]

        mock_shape = Mock()
        mock_shape.has_table = False
        mock_shape.has_text_frame = True
        mock_shape.text_frame = mock_text_frame

        mock_slide = Mock()
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_prs.core_properties.title = "프레젠테이션 제목"
        mock_prs.core_properties.author = "홍길동"
        mock_prs.core_properties.created = datetime(2024, 3, 20, 9, 0, 0)

        parser = PPTXParser()

        with patch("pptx.Presentation", return_value=mock_prs):
            doc = parser.parse(pptx_file)

            assert doc.title == "프레젠테이션 제목"
            assert doc.metadata["author"] == "홍길동"
            assert doc.metadata["date"] == "2024-03-20"
            assert doc.metadata["slide_count"] == 1
