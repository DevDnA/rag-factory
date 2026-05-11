"""PDF 파서의 fixture 기반 테스트 — 실제 PDF 파일을 생성하여 파싱 결과를 검증합니다.

Windows 맑은고딕(malgun.ttf) 폰트를 사용하여 한국어 텍스트가 포함된
실제 PDF를 생성하고, PDFParser가 정확히 추출하는지 검증합니다.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from slm_factory.parsers.pdf import PDFParser, _clean_page_numbers, _table_to_markdown

# 맑은고딕 폰트 경로 (WSL 환경에서 Windows 폰트 참조)
_MALGUN_FONT = "/mnt/c/Windows/Fonts/malgun.ttf"

pytestmark = pytest.mark.skipif(
    not Path(_MALGUN_FONT).exists(),
    reason="맑은고딕 폰트 없음 (WSL 환경 전용 테스트)",
)


# ---------------------------------------------------------------------------
# fixture: 실제 PDF 파일 생성 헬퍼
# ---------------------------------------------------------------------------


def _create_pdf(path: Path, pages: list[str], *, metadata: dict | None = None) -> Path:
    """주어진 텍스트 페이지로 실제 PDF 파일을 생성합니다.

    매개변수
    ----------
    path:
        생성할 PDF 파일 경로입니다.
    pages:
        각 페이지에 삽입할 텍스트 목록입니다.
    metadata:
        PDF 메타데이터 (title, author 등)입니다.
    """
    doc = fitz.open()  # 빈 PDF 생성
    for text in pages:
        page = doc.new_page()
        # 맑은고딕 폰트로 한국어 텍스트 삽입
        page.insert_text(
            (72, 72), text, fontsize=12,
            fontname="malgun", fontfile=_MALGUN_FONT,
        )

    if metadata:
        doc.set_metadata(metadata)

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def simple_pdf(tmp_path: Path) -> Path:
    """단순 텍스트 1페이지 PDF를 생성합니다."""
    return _create_pdf(
        tmp_path / "simple.pdf",
        ["안녕하세요. 이것은 테스트 PDF 문서입니다."],
    )


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    """여러 페이지 텍스트를 포함하는 PDF를 생성합니다."""
    return _create_pdf(
        tmp_path / "multi_page.pdf",
        [
            "첫 번째 페이지의 내용입니다.",
            "두 번째 페이지에는 다른 내용이 있습니다.",
            "세 번째 페이지는 마지막입니다.",
        ],
    )


@pytest.fixture
def pdf_with_metadata(tmp_path: Path) -> Path:
    """메타데이터가 포함된 PDF를 생성합니다."""
    return _create_pdf(
        tmp_path / "with_meta.pdf",
        ["메타데이터가 있는 문서입니다."],
        metadata={
            "title": "테스트 문서 제목",
            "author": "홍길동",
            "subject": "테스트용 PDF",
        },
    )


@pytest.fixture
def pdf_with_date_filename(tmp_path: Path) -> Path:
    """파일명에 YYMMDD 날짜가 포함된 PDF를 생성합니다."""
    return _create_pdf(
        tmp_path / "report_240115_v2.pdf",
        ["날짜가 파일명에 포함된 문서입니다."],
    )


@pytest.fixture
def pdf_with_page_numbers(tmp_path: Path) -> Path:
    """페이지 번호 패턴이 포함된 텍스트를 가진 PDF를 생성합니다."""
    return _create_pdf(
        tmp_path / "page_nums.pdf",
        [
            "첫 번째 페이지 내용\n- 1 -",
            "두 번째 페이지 내용\nPage 2",
        ],
    )


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """텍스트가 없는 빈 PDF를 생성합니다."""
    doc = fitz.open()
    doc.new_page()  # 빈 페이지
    path = tmp_path / "empty.pdf"
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# PDFParser 기본 동작 테스트
# ---------------------------------------------------------------------------


class TestPDFParserBasic:
    """PDFParser의 기본 파싱 기능을 검증합니다."""

    def test_parse_simple_pdf(self, simple_pdf: Path):
        """단순 PDF에서 한국어 텍스트가 정상적으로 추출되는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(simple_pdf)

        assert result.doc_id == "simple.pdf"
        assert "테스트 PDF 문서" in result.content
        assert result.title == "simple"  # 메타데이터 title 없으면 파일명

    def test_parse_multi_page(self, multi_page_pdf: Path):
        """여러 페이지의 한국어 텍스트가 모두 추출되는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(multi_page_pdf)

        assert "첫 번째 페이지" in result.content
        assert "두 번째 페이지" in result.content
        assert "세 번째 페이지" in result.content

    def test_parse_returns_parsed_document(self, simple_pdf: Path):
        """반환 타입이 ParsedDocument인지 확인합니다."""
        from slm_factory.models import ParsedDocument

        parser = PDFParser()
        result = parser.parse(simple_pdf)

        assert isinstance(result, ParsedDocument)
        assert isinstance(result.content, str)
        assert isinstance(result.tables, list)
        assert isinstance(result.metadata, dict)

    def test_parse_empty_pdf(self, empty_pdf: Path):
        """빈 PDF를 파싱해도 오류 없이 빈 content를 반환합니다."""
        parser = PDFParser()
        result = parser.parse(empty_pdf)

        assert result.content == ""
        assert result.doc_id == "empty.pdf"


# ---------------------------------------------------------------------------
# 메타데이터 추출 테스트
# ---------------------------------------------------------------------------


class TestPDFParserMetadata:
    """PDF 메타데이터 추출 기능을 검증합니다."""

    def test_metadata_extracted(self, pdf_with_metadata: Path):
        """PDF 메타데이터(author, subject)가 정상 추출되는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(pdf_with_metadata)

        assert result.metadata["author"] == "홍길동"
        assert result.metadata["subject"] == "테스트용 PDF"

    def test_title_from_metadata(self, pdf_with_metadata: Path):
        """PDF 메타데이터의 title이 문서 제목으로 사용되는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(pdf_with_metadata)

        assert result.title == "테스트 문서 제목"

    def test_title_fallback_to_filename(self, simple_pdf: Path):
        """PDF 메타데이터에 title이 없으면 파일명(stem)을 사용하는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(simple_pdf)

        assert result.title == "simple"

    def test_page_count_in_metadata(self, multi_page_pdf: Path):
        """page_count가 메타데이터에 정확히 기록되는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(multi_page_pdf)

        assert result.metadata["page_count"] == 3

    def test_date_from_filename(self, pdf_with_date_filename: Path):
        """파일명의 YYMMDD 패턴에서 날짜가 추출되는지 확인합니다."""
        parser = PDFParser()
        result = parser.parse(pdf_with_date_filename)

        assert result.metadata["date"] == "2024-01-15"


# ---------------------------------------------------------------------------
# 페이지 번호 정리 테스트
# ---------------------------------------------------------------------------


class TestCleanPageNumbers:
    """_clean_page_numbers 함수의 페이지 번호 제거 로직을 검증합니다."""

    def test_removes_dash_page_numbers(self):
        """'- 1 -' 형식의 페이지 번호를 제거합니다."""
        text = "본문 내용\n- 1 -\n다음 내용"
        result = _clean_page_numbers(text)
        assert "- 1 -" not in result
        assert "본문 내용" in result
        assert "다음 내용" in result

    def test_removes_em_dash_page_numbers(self):
        """'— 12 —' 형식의 페이지 번호를 제거합니다."""
        text = "텍스트\n— 12 —\n추가 텍스트"
        result = _clean_page_numbers(text)
        assert "— 12 —" not in result

    def test_removes_page_keyword(self):
        """'Page 3' 형식의 페이지 번호를 제거합니다."""
        text = "본문\nPage 3\n다음"
        result = _clean_page_numbers(text)
        assert "Page 3" not in result

    def test_removes_standalone_numbers(self):
        """독립형 숫자 라인을 제거합니다."""
        text = "본문\n42\n다음"
        result = _clean_page_numbers(text)
        assert "\n42\n" not in result

    def test_preserves_inline_numbers(self):
        """문장 내부의 숫자는 제거하지 않습니다."""
        text = "총 42개의 항목이 있습니다."
        result = _clean_page_numbers(text)
        assert "42" in result


# ---------------------------------------------------------------------------
# 에러 핸들링 테스트
# ---------------------------------------------------------------------------


class TestPDFParserErrors:
    """PDFParser의 에러 처리를 검증합니다."""

    def test_file_not_found(self, tmp_path: Path):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시킵니다."""
        parser = PDFParser()
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            parser.parse(tmp_path / "nonexistent.pdf")

    def test_invalid_pdf_raises_runtime_error(self, tmp_path: Path):
        """유효하지 않은 PDF 파일에 대해 RuntimeError를 발생시킵니다."""
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"this is not a valid pdf")

        parser = PDFParser()
        with pytest.raises(RuntimeError, match="PDF 파일을 열 수 없습니다"):
            parser.parse(bad_pdf)


# ---------------------------------------------------------------------------
# 파서 등록 및 확장자 테스트
# ---------------------------------------------------------------------------


class TestPDFParserRegistration:
    """PDFParser의 확장자 매칭과 can_parse를 검증합니다."""

    def test_extensions(self):
        """지원 확장자가 .pdf인지 확인합니다."""
        assert PDFParser.extensions == [".pdf"]

    def test_can_parse_pdf(self, simple_pdf: Path):
        """PDF 파일에 대해 can_parse가 True를 반환합니다."""
        parser = PDFParser()
        assert parser.can_parse(simple_pdf) is True

    def test_cannot_parse_txt(self, tmp_path: Path):
        """TXT 파일에 대해 can_parse가 False를 반환합니다."""
        txt = tmp_path / "file.txt"
        txt.write_text("text")
        parser = PDFParser()
        assert parser.can_parse(txt) is False


# ---------------------------------------------------------------------------
# pymupdf4llm 경로 mock 테스트 (폰트 불필요)
# ---------------------------------------------------------------------------


class TestPDFParserPymupdf4llm:
    """pymupdf4llm 설치 시 마크다운 추출 경로를 검증합니다."""

    def test_pymupdf4llm_경로_사용(self, tmp_path: Path, mocker):
        """HAS_PYMUPDF4LLM=True일 때 pymupdf4llm.to_markdown을 호출합니다."""
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.set_metadata({"title": "마크다운 테스트"})
        doc.save(str(pdf_path))
        doc.close()

        md_content = "# 제목\n\n본문 내용입니다.\n\n| 이름 | 값 |\n| --- | --- |\n| A | 1 |"
        mocker.patch("slm_factory.parsers.pdf.HAS_PYMUPDF4LLM", True)
        mocker.patch("slm_factory.parsers.pdf.pymupdf4llm")
        from slm_factory.parsers import pdf as pdf_mod
        pdf_mod.pymupdf4llm.to_markdown.return_value = md_content

        parser = PDFParser()
        result = parser.parse(pdf_path)

        pdf_mod.pymupdf4llm.to_markdown.assert_called_once()
        assert "본문 내용" in result.content
        assert result.title == "마크다운 테스트"
        assert len(result.tables) == 1

    def test_pymupdf4llm_실패시_fitz_폴백(self, tmp_path: Path, mocker):
        """pymupdf4llm.to_markdown 실패 시 fitz 폴백을 사용합니다."""
        pdf_path = tmp_path / "fallback.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        mocker.patch("slm_factory.parsers.pdf.HAS_PYMUPDF4LLM", True)
        mocker.patch("slm_factory.parsers.pdf.pymupdf4llm")
        from slm_factory.parsers import pdf as pdf_mod
        pdf_mod.pymupdf4llm.to_markdown.side_effect = RuntimeError("OCR 실패")

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert isinstance(result.content, str)
        assert result.doc_id == "fallback.pdf"

    def test_pymupdf4llm_미설치시_fitz_사용(self, tmp_path: Path, mocker):
        """HAS_PYMUPDF4LLM=False일 때 fitz 기반 추출을 사용합니다."""
        pdf_path = tmp_path / "nollm.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        mocker.patch("slm_factory.parsers.pdf.HAS_PYMUPDF4LLM", False)

        parser = PDFParser()
        result = parser.parse(pdf_path)

        assert isinstance(result.content, str)
        assert result.doc_id == "nollm.pdf"


class TestExtractTablesFromMarkdown:
    """_extract_tables_from_markdown 함수를 검증합니다."""

    def test_마크다운_표_추출(self):
        """마크다운 텍스트에서 표 블록을 정확히 추출합니다."""
        from slm_factory.parsers.pdf import _extract_tables_from_markdown

        md = "텍스트\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n추가 텍스트"
        tables = _extract_tables_from_markdown(md)
        assert len(tables) == 1
        assert "| A | B |" in tables[0]

    def test_표_없는_텍스트(self):
        """표가 없는 텍스트에서 빈 리스트를 반환합니다."""
        from slm_factory.parsers.pdf import _extract_tables_from_markdown

        tables = _extract_tables_from_markdown("그냥 텍스트입니다.")
        assert tables == []

    def test_여러_표_추출(self):
        """여러 표가 포함된 텍스트에서 모두 추출합니다."""
        from slm_factory.parsers.pdf import _extract_tables_from_markdown

        md = "| A |\n| --- |\n| 1 |\n\n텍스트\n\n| B |\n| --- |\n| 2 |"
        tables = _extract_tables_from_markdown(md)
        assert len(tables) == 2
