"""HWPX 파서의 fixture 기반 테스트 — 실제 HWPX(ZIP+XML) 파일을 생성하여 파싱 결과를 검증합니다."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from slm_factory.parsers.hwpx import HWPXParser, _table_to_markdown


# ---------------------------------------------------------------------------
# fixture: 실제 HWPX(ZIP+XML) 파일 생성 헬퍼
# ---------------------------------------------------------------------------

# HWPX 형식의 기본 XML 구조
_SECTION_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<hs:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"
         xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"
         xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app">
{body}
</hs:sec>
"""


def _make_paragraph(text: str) -> str:
    """hp:p > hp:run > hp:t 구조의 단락 XML을 생성합니다."""
    return f"  <hp:p><hp:run><hp:t>{text}</hp:t></hp:run></hp:p>"


def _make_table(rows: list[list[str]]) -> str:
    """hp:tbl > hp:tr > hp:tc 구조의 표 XML을 생성합니다.

    매개변수
    ----------
    rows:
        2차원 문자열 리스트입니다. 첫 행은 헤더로 사용됩니다.
    """
    lines = ["  <hp:tbl>"]
    for row in rows:
        lines.append("    <hp:tr>")
        for cell in row:
            lines.append(
                f"      <hp:tc><hp:p><hp:run><hp:t>{cell}</hp:t></hp:run></hp:p></hp:tc>"
            )
        lines.append("    </hp:tr>")
    lines.append("  </hp:tbl>")
    return "\n".join(lines)


def _create_hwpx(path: Path, body_xml: str) -> Path:
    """body_xml을 Contents/section0.xml에 넣은 HWPX(ZIP) 파일을 생성합니다."""
    full_xml = _SECTION_XML_TEMPLATE.format(body=body_xml)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Contents/section0.xml", full_xml)
    return path


def _create_multi_section_hwpx(path: Path, sections: list[str]) -> Path:
    """여러 섹션 XML을 가진 멀티섹션 HWPX(ZIP) 파일을 생성합니다.

    매개변수
    ----------
    path:
        생성할 HWPX 파일 경로.
    sections:
        각 섹션의 body XML 리스트.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, body_xml in enumerate(sections):
            full_xml = _SECTION_XML_TEMPLATE.format(body=body_xml)
            zf.writestr(f"Contents/section{idx}.xml", full_xml)
    return path


# ---------------------------------------------------------------------------
# pytest fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_hwpx(tmp_path: Path) -> Path:
    """단순 텍스트 단락이 포함된 HWPX 파일을 생성합니다."""
    body = "\n".join(
        [
            _make_paragraph("안녕하세요. HWPX 테스트 문서입니다."),
            _make_paragraph("두 번째 단락의 내용입니다."),
        ]
    )
    return _create_hwpx(tmp_path / "simple.hwpx", body)


@pytest.fixture
def hwpx_with_table(tmp_path: Path) -> Path:
    """텍스트와 표가 포함된 HWPX 파일을 생성합니다."""
    body = "\n".join(
        [
            _make_paragraph("표가 포함된 문서입니다."),
            _make_table(
                [
                    ["이름", "나이", "직업"],
                    ["홍길동", "30", "개발자"],
                    ["김철수", "25", "디자이너"],
                ]
            ),
            _make_paragraph("표 이후의 텍스트입니다."),
        ]
    )
    return _create_hwpx(tmp_path / "with_table.hwpx", body)


@pytest.fixture
def hwpx_with_date_filename(tmp_path: Path) -> Path:
    """파일명에 YYMMDD 날짜가 포함된 HWPX 파일을 생성합니다."""
    body = _make_paragraph("날짜 파일명 테스트입니다.")
    return _create_hwpx(tmp_path / "memo_250115_final.hwpx", body)


@pytest.fixture
def hwpx_multiple_paragraphs(tmp_path: Path) -> Path:
    """여러 단락이 포함된 HWPX 파일을 생성합니다."""
    body = "\n".join(
        [
            _make_paragraph("첫 번째 단락입니다."),
            _make_paragraph("두 번째 단락입니다."),
            _make_paragraph("세 번째 단락입니다."),
            _make_paragraph("네 번째 단락입니다."),
        ]
    )
    return _create_hwpx(tmp_path / "multi_para.hwpx", body)


@pytest.fixture
def hwpx_empty_content(tmp_path: Path) -> Path:
    """텍스트가 없는 빈 HWPX 파일을 생성합니다."""
    return _create_hwpx(tmp_path / "empty.hwpx", "")


@pytest.fixture
def hwpx_multi_run_paragraph(tmp_path: Path) -> Path:
    """하나의 단락에 여러 hp:t 태그가 있는 HWPX 파일을 생성합니다."""
    body = (
        "  <hp:p>"
        "<hp:run><hp:t>첫 번째 </hp:t></hp:run>"
        "<hp:run><hp:t>두 번째 </hp:t></hp:run>"
        "<hp:run><hp:t>세 번째</hp:t></hp:run>"
        "</hp:p>"
    )
    return _create_hwpx(tmp_path / "multi_run.hwpx", body)


@pytest.fixture
def hwpx_multiple_tables(tmp_path: Path) -> Path:
    """여러 개의 표가 포함된 HWPX 파일을 생성합니다."""
    body = "\n".join(
        [
            _make_paragraph("첫 번째 표:"),
            _make_table(
                [
                    ["항목", "수량"],
                    ["사과", "10"],
                ]
            ),
            _make_paragraph("두 번째 표:"),
            _make_table(
                [
                    ["과목", "점수"],
                    ["수학", "90"],
                    ["영어", "85"],
                ]
            ),
        ]
    )
    return _create_hwpx(tmp_path / "multi_tables.hwpx", body)


# ---------------------------------------------------------------------------
# HWPXParser 기본 동작 테스트
# ---------------------------------------------------------------------------


class TestHWPXParserBasic:
    """HWPXParser의 기본 파싱 기능을 검증합니다."""

    def test_parse_simple_hwpx(self, simple_hwpx: Path):
        """단순 HWPX에서 텍스트가 정상적으로 추출되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(simple_hwpx)

        assert result.doc_id == "simple.hwpx"
        assert "HWPX 테스트 문서" in result.content
        assert "두 번째 단락" in result.content

    def test_parse_returns_parsed_document(self, simple_hwpx: Path):
        """반환 타입이 ParsedDocument인지 확인합니다."""
        from slm_factory.models import ParsedDocument

        parser = HWPXParser()
        result = parser.parse(simple_hwpx)

        assert isinstance(result, ParsedDocument)
        assert isinstance(result.content, str)
        assert isinstance(result.tables, list)
        assert isinstance(result.metadata, dict)

    def test_parse_multiple_paragraphs(self, hwpx_multiple_paragraphs: Path):
        """여러 단락이 모두 추출되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_multiple_paragraphs)

        assert "첫 번째 단락" in result.content
        assert "두 번째 단락" in result.content
        assert "세 번째 단락" in result.content
        assert "네 번째 단락" in result.content

    def test_title_is_filename_stem(self, simple_hwpx: Path):
        """제목이 파일명(stem)으로 설정되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(simple_hwpx)

        assert result.title == "simple"

    def test_parse_empty_content(self, hwpx_empty_content: Path):
        """빈 HWPX를 파싱해도 오류 없이 빈 content를 반환합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_empty_content)

        assert result.content == ""

    def test_multi_run_concatenation(self, hwpx_multi_run_paragraph: Path):
        """하나의 단락 내 여러 hp:t 태그의 텍스트가 연결되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_multi_run_paragraph)

        assert "첫 번째" in result.content
        assert "두 번째" in result.content
        assert "세 번째" in result.content


# ---------------------------------------------------------------------------
# 표 추출 테스트
# ---------------------------------------------------------------------------


class TestHWPXParserTables:
    """HWPX 표 추출 기능을 검증합니다."""

    def test_table_extracted(self, hwpx_with_table: Path):
        """표가 마크다운 형식으로 추출되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_with_table)

        assert len(result.tables) == 1
        table_md = result.tables[0]
        assert "이름" in table_md
        assert "나이" in table_md
        assert "직업" in table_md
        assert "홍길동" in table_md
        assert "30" in table_md

    def test_table_markdown_format(self, hwpx_with_table: Path):
        """추출된 표가 올바른 마크다운 표 형식인지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_with_table)

        table_md = result.tables[0]
        lines = table_md.strip().split("\n")

        # 첫 행: 헤더
        assert lines[0].startswith("|")
        assert lines[0].endswith("|")
        # 두 번째 행: 구분선
        assert "---" in lines[1]
        # 데이터 행
        assert len(lines) == 4  # 헤더 + 구분선 + 2개 데이터 행

    def test_multiple_tables(self, hwpx_multiple_tables: Path):
        """여러 개의 표가 모두 추출되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_multiple_tables)

        assert len(result.tables) == 2
        # 첫 번째 표
        assert "항목" in result.tables[0]
        assert "사과" in result.tables[0]
        # 두 번째 표
        assert "과목" in result.tables[1]
        assert "수학" in result.tables[1]

    def test_text_around_table(self, hwpx_with_table: Path):
        """표 전후의 텍스트도 정상 추출되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_with_table)

        assert "표가 포함된 문서" in result.content
        assert "표 이후의 텍스트" in result.content


# ---------------------------------------------------------------------------
# _table_to_markdown 단위 테스트
# ---------------------------------------------------------------------------


class TestTableToMarkdown:
    """_table_to_markdown 헬퍼 함수를 직접 검증합니다."""

    def test_basic_table(self):
        """기본 2x2 표 변환을 확인합니다."""
        from bs4 import BeautifulSoup

        xml = """
        <hp:tbl xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
            <hp:tr>
                <hp:tc><hp:p><hp:t>A</hp:t></hp:p></hp:tc>
                <hp:tc><hp:p><hp:t>B</hp:t></hp:p></hp:tc>
            </hp:tr>
            <hp:tr>
                <hp:tc><hp:p><hp:t>1</hp:t></hp:p></hp:tc>
                <hp:tc><hp:p><hp:t>2</hp:t></hp:p></hp:tc>
            </hp:tr>
        </hp:tbl>
        """
        soup = BeautifulSoup(xml, "xml")
        table_el = soup.find("hp:tbl")
        result = _table_to_markdown(table_el)

        assert "| A | B |" in result
        assert "| 1 | 2 |" in result
        assert "---" in result

    def test_empty_table(self):
        """행이 없는 빈 표에 대해 빈 문자열을 반환합니다."""
        from bs4 import BeautifulSoup

        xml = (
            '<hp:tbl xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"></hp:tbl>'
        )
        soup = BeautifulSoup(xml, "xml")
        table_el = soup.find("hp:tbl")
        result = _table_to_markdown(table_el)

        assert result == ""


# ---------------------------------------------------------------------------
# 메타데이터 테스트
# ---------------------------------------------------------------------------


class TestHWPXParserMetadata:
    """HWPX 메타데이터 추출 기능을 검증합니다."""

    def test_date_from_filename(self, hwpx_with_date_filename: Path):
        """파일명의 YYMMDD 패턴에서 날짜가 추출되는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(hwpx_with_date_filename)

        assert result.metadata["date"] == "2025-01-15"

    def test_no_date_in_metadata(self, simple_hwpx: Path):
        """파일명에 날짜가 없으면 metadata에 date 키가 없는지 확인합니다."""
        parser = HWPXParser()
        result = parser.parse(simple_hwpx)

        assert "date" not in result.metadata


# ---------------------------------------------------------------------------
# 에러 핸들링 테스트
# ---------------------------------------------------------------------------


class TestHWPXParserErrors:
    """HWPXParser의 에러 처리를 검증합니다."""

    def test_file_not_found(self, tmp_path: Path):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시킵니다."""
        parser = HWPXParser()
        with pytest.raises(FileNotFoundError, match="HWPX not found"):
            parser.parse(tmp_path / "nonexistent.hwpx")

    def test_missing_section_xml(self, tmp_path: Path):
        """Contents/section*.xml이 없는 ZIP에 대해 RuntimeError를 발생시킵니다."""
        bad_hwpx = tmp_path / "no_section.hwpx"
        with zipfile.ZipFile(bad_hwpx, "w") as zf:
            zf.writestr("dummy.txt", "not a valid hwpx")

        parser = HWPXParser()
        with pytest.raises(RuntimeError, match="올바르지 않습니다"):
            parser.parse(bad_hwpx)

    def test_invalid_zip(self, tmp_path: Path):
        """유효하지 않은 ZIP 파일에 대해 RuntimeError를 발생시킵니다."""
        bad_file = tmp_path / "corrupt.hwpx"
        bad_file.write_bytes(b"this is not a zip file")

        parser = HWPXParser()
        with pytest.raises(RuntimeError, match="유효한 HWPX"):
            parser.parse(bad_file)


# ---------------------------------------------------------------------------
# 파서 등록 및 확장자 테스트
# ---------------------------------------------------------------------------


class TestHWPXParserRegistration:
    """HWPXParser의 확장자 매칭과 can_parse를 검증합니다."""

    def test_extensions(self):
        """지원 확장자가 .hwpx인지 확인합니다."""
        assert HWPXParser.extensions == [".hwpx"]

    def test_can_parse_hwpx(self, simple_hwpx: Path):
        """HWPX 파일에 대해 can_parse가 True를 반환합니다."""
        parser = HWPXParser()
        assert parser.can_parse(simple_hwpx) is True

    def test_cannot_parse_pdf(self, tmp_path: Path):
        """PDF 파일에 대해 can_parse가 False를 반환합니다."""
        pdf = tmp_path / "file.pdf"
        pdf.write_bytes(b"fake")
        parser = HWPXParser()
        assert parser.can_parse(pdf) is False


# ---------------------------------------------------------------------------
# 멀티섹션 HWPX 테스트
# ---------------------------------------------------------------------------


class TestMultiSectionHWPX:
    """멀티섹션 HWPX 파일의 파싱을 검증합니다."""

    def test_두_섹션_텍스트_병합(self, tmp_path: Path):
        """section0.xml과 section1.xml의 텍스트가 순서대로 병합됩니다."""
        hwpx = _create_multi_section_hwpx(
            tmp_path / "multi.hwpx",
            [
                _make_paragraph("섹션0의 내용입니다."),
                _make_paragraph("섹션1의 내용입니다."),
            ],
        )
        parser = HWPXParser()
        result = parser.parse(hwpx)

        assert "내용입니다" in result.content
        assert result.content.index("섹션") < result.content.rindex("섹션")

    def test_세_섹션_표_병합(self, tmp_path: Path):
        """여러 섹션에 걸친 표가 모두 추출됩니다."""
        table1 = _make_table([["A", "B"], ["1", "2"]])
        table2 = _make_table([["C", "D"], ["3", "4"]])
        hwpx = _create_multi_section_hwpx(
            tmp_path / "tables.hwpx",
            [
                _make_paragraph("첫 섹션") + "\n" + table1,
                _make_paragraph("둘째 섹션") + "\n" + table2,
            ],
        )
        parser = HWPXParser()
        result = parser.parse(hwpx)

        assert len(result.tables) == 2

    def test_단일_섹션_하위호환(self, tmp_path: Path):
        """section0.xml만 있는 기존 HWPX 파일이 정상 파싱됩니다."""
        hwpx = _create_hwpx(
            tmp_path / "single.hwpx",
            _make_paragraph("단일 섹션 문서입니다."),
        )
        parser = HWPXParser()
        result = parser.parse(hwpx)

        assert "단일 섹션 문서입니다" in result.content

    def test_섹션_없는_HWPX_에러(self, tmp_path: Path):
        """Contents/section*.xml이 없는 HWPX는 RuntimeError를 발생시킵니다."""
        broken = tmp_path / "broken.hwpx"
        with zipfile.ZipFile(broken, "w") as zf:
            zf.writestr("Contents/other.xml", "<doc/>")

        parser = HWPXParser()
        with pytest.raises(RuntimeError, match="올바르지 않습니다"):
            parser.parse(broken)
