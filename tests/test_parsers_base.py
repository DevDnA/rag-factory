"""파서 기본 모듈(parsers/base.py)의 단위 테스트입니다."""

from pathlib import Path

from slm_factory.models import ParsedDocument
from slm_factory.parsers.base import BaseParser, ParserRegistry, extract_date_from_filename, rows_to_markdown


# ---------------------------------------------------------------------------
# extract_date_from_filename
# ---------------------------------------------------------------------------


class TestExtractDateFromFilename:
    """extract_date_from_filename 함수의 테스트입니다."""

    def test_유효한_날짜_추출(self):
        """YYMMDD 패턴이 있는 파일명에서 YYYY-MM-DD 형식으로 변환하는지 확인합니다."""
        result = extract_date_from_filename("report_240115_v2")
        assert result == "2024-01-15"

    def test_날짜_없는_파일명(self):
        """날짜 패턴이 없는 파일명에서 None을 반환하는지 확인합니다."""
        result = extract_date_from_filename("document_final")
        assert result is None

    def test_잘못된_날짜_13월(self):
        """13월 같은 잘못된 날짜 패턴은 None을 반환하는지 확인합니다."""
        result = extract_date_from_filename("report_241315")
        assert result is None

    def test_연말_날짜(self):
        """12월 31일 같은 경계 날짜도 올바르게 추출하는지 확인합니다."""
        result = extract_date_from_filename("data_251231")
        assert result == "2025-12-31"


# ---------------------------------------------------------------------------
# BaseParser 서브클래스 테스트용 더미 파서
# ---------------------------------------------------------------------------


class _DummyParser(BaseParser):
    """테스트를 위한 더미 파서입니다."""

    extensions = [".dummy", ".test"]

    def parse(self, path: Path) -> ParsedDocument:
        """더미 ParsedDocument를 반환합니다."""
        return ParsedDocument(
            doc_id=path.name,
            title=path.stem,
            content="더미 내용",
        )


# ---------------------------------------------------------------------------
# BaseParser
# ---------------------------------------------------------------------------


class TestBaseParser:
    """BaseParser ABC의 can_parse 메서드 테스트입니다."""

    def test_지원_확장자_True(self):
        """지원하는 확장자의 파일 경로에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = _DummyParser()
        assert parser.can_parse(Path("file.dummy")) is True
        assert parser.can_parse(Path("file.test")) is True

    def test_미지원_확장자_False(self):
        """지원하지 않는 확장자의 파일 경로에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = _DummyParser()
        assert parser.can_parse(Path("file.pdf")) is False
        assert parser.can_parse(Path("file.txt")) is False


# ---------------------------------------------------------------------------
# ParserRegistry
# ---------------------------------------------------------------------------


class TestParserRegistry:
    """ParserRegistry 클래스의 테스트입니다."""

    def test_register_파서_등록(self):
        """파서 클래스를 등록하면 _parsers 리스트에 추가되는지 확인합니다."""
        registry = ParserRegistry()
        registry.register(_DummyParser)
        assert len(registry._parsers) == 1

    def test_get_parser_등록된_확장자(self):
        """등록된 확장자의 파일 경로에 대해 파서 인스턴스를 반환하는지 확인합니다."""
        registry = ParserRegistry()
        registry.register(_DummyParser)
        parser = registry.get_parser(Path("file.dummy"))
        assert isinstance(parser, _DummyParser)

    def test_get_parser_미등록_확장자(self):
        """미등록 확장자의 파일 경로에 대해 None을 반환하는지 확인합니다."""
        registry = ParserRegistry()
        registry.register(_DummyParser)
        parser = registry.get_parser(Path("file.xyz"))
        assert parser is None

    def test_parse_directory_빈_디렉토리(self, tmp_path):
        """빈 디렉토리에서 빈 리스트를 반환하는지 확인합니다."""
        registry = ParserRegistry()
        registry.register(_DummyParser)
        result = registry.parse_directory(tmp_path)
        assert result == []

    def test_parse_directory_존재하지_않는_디렉토리(self):
        """존재하지 않는 디렉토리에서 빈 리스트를 반환하는지 확인합니다."""
        registry = ParserRegistry()
        result = registry.parse_directory(Path("/nonexistent/path"))
        assert result == []

    def test_parse_directory_파일이_있는_디렉토리(self, tmp_path):
        """파서가 등록된 확장자의 파일이 있는 디렉토리에서 문서를 파싱하는지 확인합니다."""
        # .dummy 확장자의 파일 생성
        (tmp_path / "doc1.dummy").write_text("내용1", encoding="utf-8")
        (tmp_path / "doc2.dummy").write_text("내용2", encoding="utf-8")
        (tmp_path / "ignored.xyz").write_text("무시됨", encoding="utf-8")

        registry = ParserRegistry()
        registry.register(_DummyParser)
        result = registry.parse_directory(tmp_path)

        assert len(result) == 2
        doc_ids = {doc.doc_id for doc in result}
        assert "doc1.dummy" in doc_ids
        assert "doc2.dummy" in doc_ids

    def test_parse_directory_formats_필터(self, tmp_path):
        """formats 인자로 특정 확장자만 필터링하는지 확인합니다."""
        (tmp_path / "doc1.dummy").write_text("내용1", encoding="utf-8")
        (tmp_path / "doc2.test").write_text("내용2", encoding="utf-8")

        registry = ParserRegistry()
        registry.register(_DummyParser)
        result = registry.parse_directory(tmp_path, formats=[".dummy"])

        assert len(result) == 1
        assert result[0].doc_id == "doc1.dummy"

    def test_parse_directory_files_필터(self, tmp_path):
        """files 인자로 특정 파일만 파싱하는지 확인합니다."""
        file1 = tmp_path / "doc1.dummy"
        file2 = tmp_path / "doc2.dummy"
        file1.write_text("내용1", encoding="utf-8")
        file2.write_text("내용2", encoding="utf-8")

        registry = ParserRegistry()
        registry.register(_DummyParser)
        result = registry.parse_directory(tmp_path, files=[file1])

        assert len(result) == 1
        assert result[0].doc_id == "doc1.dummy"


# ---------------------------------------------------------------------------
# rows_to_markdown
# ---------------------------------------------------------------------------


class TestRowsToMarkdown:
    """rows_to_markdown 함수의 테스트입니다."""

    def test_빈_리스트(self):
        """빈 리스트가 주어지면 빈 문자열을 반환하는지 확인합니다."""
        assert rows_to_markdown([]) == ""

    def test_헤더만_있는_빈_행(self):
        """헤더 행의 셀이 모두 빈 리스트이면 빈 문자열을 반환하는지 확인합니다."""
        assert rows_to_markdown([[]]) == ""

    def test_단일_행_헤더만(self):
        """헤더만 있고 데이터 행이 없는 경우 올바른 마크다운을 반환하는지 확인합니다."""
        result = rows_to_markdown([["이름", "나이"]])
        lines = result.split("\n")
        assert len(lines) == 2
        assert "이름" in lines[0]
        assert "나이" in lines[0]
        assert "---" in lines[1]

    def test_여러_행(self):
        """헤더와 데이터 행이 모두 있는 경우 올바른 마크다운 테이블을 생성하는지 확인합니다."""
        result = rows_to_markdown([
            ["A", "B"],
            ["1", "2"],
            ["3", "4"],
        ])
        lines = result.split("\n")
        assert len(lines) == 4  # 헤더 + 구분선 + 2 데이터
        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | 2 |"
        assert lines[3] == "| 3 | 4 |"

    def test_열_수_부족_패딩(self):
        """데이터 행의 열 수가 헤더보다 적으면 빈 문자열로 패딩하는지 확인합니다."""
        result = rows_to_markdown([
            ["A", "B", "C"],
            ["1"],
        ])
        lines = result.split("\n")
        assert lines[2] == "| 1 |  |  |"

    def test_열_수_초과_자름(self):
        """데이터 행의 열 수가 헤더보다 많으면 헤더 열 수에 맞게 자르는지 확인합니다."""
        result = rows_to_markdown([
            ["A", "B"],
            ["1", "2", "3", "4"],
        ])
        lines = result.split("\n")
        assert lines[2] == "| 1 | 2 |"
