"""HTML 파서(parsers/html.py)의 단위 테스트입니다."""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from slm_factory.parsers.html import HTMLParser, _table_to_markdown


# ---------------------------------------------------------------------------
# _table_to_markdown
# ---------------------------------------------------------------------------


class TestTableToMarkdown:
    """_table_to_markdown 함수의 테스트입니다."""

    def test_빈_테이블(self):
        """행이 없는 빈 테이블에서 빈 문자열을 반환하는지 확인합니다."""
        soup = BeautifulSoup("<table></table>", "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)
        assert result == ""

    def test_정상_테이블_변환(self):
        """헤더와 데이터 행이 있는 테이블을 마크다운으로 올바르게 변환하는지 확인합니다."""
        html = (
            "<table>"
            "<tr><th>이름</th><th>값</th></tr>"
            "<tr><td>A</td><td>1</td></tr>"
            "<tr><td>B</td><td>2</td></tr>"
            "</table>"
        )
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)

        assert "| 이름 | 값 |" in result
        assert "| --- | --- |" in result
        assert "| A | 1 |" in result
        assert "| B | 2 |" in result


# ---------------------------------------------------------------------------
# HTMLParser
# ---------------------------------------------------------------------------


class TestHTMLParser:
    """HTMLParser 클래스의 테스트입니다."""

    def test_can_parse_html_True(self):
        """.html 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = HTMLParser()
        assert parser.can_parse(Path("page.html")) is True

    def test_can_parse_htm_True(self):
        """.htm 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = HTMLParser()
        assert parser.can_parse(Path("page.htm")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = HTMLParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_기본_HTML(self, tmp_html_file):
        """기본 HTML 파일을 파싱하여 ParsedDocument를 반환하는지 확인합니다."""
        parser = HTMLParser()
        doc = parser.parse(tmp_html_file)

        assert doc.doc_id == "page.html"
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0

    def test_parse_제목_추출(self, tmp_html_file):
        """HTML의 <title> 태그에서 제목을 추출하는지 확인합니다."""
        parser = HTMLParser()
        doc = parser.parse(tmp_html_file)

        assert doc.title == "테스트 페이지"

    def test_parse_표_추출(self, tmp_html_file):
        """HTML의 <table>을 마크다운 표로 변환하여 tables에 저장하는지 확인합니다."""
        parser = HTMLParser()
        doc = parser.parse(tmp_html_file)

        assert len(doc.tables) >= 1
        # 마크다운 표 형식이 포함되어야 합니다
        assert "이름" in doc.tables[0]
        assert "|" in doc.tables[0]

    def test_parse_script_style_태그_제거(self, tmp_path):
        """script와 style 태그의 내용이 콘텐츠에서 제거되는지 확인합니다."""
        html_file = tmp_path / "script_test.html"
        html_file.write_text(
            "<html><body>"
            "<script>alert('위험!');</script>"
            "<style>body { color: red; }</style>"
            "<p>본문 텍스트입니다.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "alert" not in doc.content
        assert "color: red" not in doc.content
        assert "본문 텍스트" in doc.content

    def test_parse_존재하지_않는_파일(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = HTMLParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.html")

    def test_parse_h1_폴백_제목(self, tmp_path):
        """<title>이 없으면 <h1>에서 제목을 추출하는지 확인합니다."""
        html_file = tmp_path / "no_title.html"
        html_file.write_text(
            "<html><body><h1>H1 제목</h1><p>내용</p></body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert doc.title == "H1 제목"


# ---------------------------------------------------------------------------
# EUC-KR 인코딩 테스트
# ---------------------------------------------------------------------------


class TestEucKrEncoding:
    """EUC-KR 인코딩 HTML 파일의 파싱 테스트입니다."""

    def test_euc_kr_인코딩_파싱(self, tmp_path):
        """EUC-KR 인코딩된 HTML 파일을 정상적으로 파싱하는지 확인합니다."""
        html_content = "<html><head><title>테스트</title></head><body><p>한국어 본문입니다.</p></body></html>"
        html_file = tmp_path / "euckr.html"
        html_file.write_bytes(html_content.encode("euc-kr"))

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "한국어 본문" in doc.content
        assert doc.title == "테스트"

    def test_cp949_인코딩_파싱(self, tmp_path):
        """CP949 인코딩된 HTML 파일을 정상적으로 파싱하는지 확인합니다."""
        html_content = "<html><body><p>CP949로 인코딩된 텍스트입니다.</p></body></html>"
        html_file = tmp_path / "cp949.html"
        html_file.write_bytes(html_content.encode("cp949"))

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "CP949로 인코딩된" in doc.content


# ---------------------------------------------------------------------------
# Heading 마크다운 변환 테스트
# ---------------------------------------------------------------------------


class TestHeadingPreservation:
    """HTML heading 태그의 마크다운 변환 테스트입니다."""

    def test_h1_마크다운_변환(self, tmp_path):
        """h1 태그가 마크다운 # 형식으로 변환되는지 확인합니다."""
        html_file = tmp_path / "heading.html"
        html_file.write_text(
            "<html><body><h1>제목</h1><p>본문</p></body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "# 제목" in doc.content

    def test_다단계_heading_변환(self, tmp_path):
        """h1~h3 태그가 각각 올바른 마크다운 레벨로 변환되는지 확인합니다."""
        html_file = tmp_path / "multi_heading.html"
        html_file.write_text(
            "<html><body>"
            "<h1>대제목</h1>"
            "<h2>중제목</h2>"
            "<h3>소제목</h3>"
            "<p>본문 내용</p>"
            "</body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "# 대제목" in doc.content
        assert "## 중제목" in doc.content
        assert "### 소제목" in doc.content

    def test_heading_없는_HTML(self, tmp_path):
        """heading 태그가 없는 HTML에서도 정상 파싱되는지 확인합니다."""
        html_file = tmp_path / "no_heading.html"
        html_file.write_text(
            "<html><body><p>본문만 있는 문서입니다.</p></body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "본문만 있는 문서" in doc.content
