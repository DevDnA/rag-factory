"""DOCX (Word) 문서 파서입니다."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..models import ParsedDocument
from ..utils import get_logger
from .base import BaseParser, extract_date_from_filename, rows_to_markdown

logger = get_logger("parsers.docx")


class DOCXParser(BaseParser):
    """DOCX 문서를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".docx"]

    def can_parse_content(self, path: Path) -> bool:
        import zipfile

        try:
            if not zipfile.is_zipfile(str(path)):
                return False
            with zipfile.ZipFile(path, "r") as zf:
                return "word/document.xml" in zf.namelist()
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """DOCX 파일을 파싱하여 ParsedDocument를 반환합니다.

        매개변수
        ----------
        path:
            DOCX 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        ImportError
            python-docx가 설치되지 않았으면 발생합니다.
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            DOCX 파일을 파싱할 수 없으면 발생합니다.
        """
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "python-docx가 설치되지 않았습니다. uv sync --extra docx 로 설치하세요."
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        # ------------------------------------------------------------------
        # DOCX 파일 로드
        # ------------------------------------------------------------------
        try:
            doc = Document(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"DOCX 파일을 파싱할 수 없습니다: {path}\n"
                f"원인: 파일이 손상되었거나 python-docx가 지원하지 않는 형식일 수 있습니다.\n"
                f"해결: Word에서 다시 저장하거나 PDF로 변환하세요. "
                f"python-docx 설치 확인: uv sync --extra docx"
            ) from exc

        # ------------------------------------------------------------------
        # 텍스트 추출
        # ------------------------------------------------------------------
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # 헤딩 스타일 처리
                if (
                    para.style
                    and hasattr(para.style, "name")
                    and isinstance(para.style.name, str)
                ):
                    if para.style.name.startswith("Heading"):
                        # Heading 1, Heading 2, Heading 10 등에서 숫자 추출
                        level = 1
                        level_part = para.style.name.replace("Heading", "").strip()
                        if level_part.isdigit():
                            level = max(1, min(int(level_part), 6))
                        paragraphs.append(f"{'#' * level} {text}")
                    else:
                        paragraphs.append(text)
                else:
                    paragraphs.append(text)

        content = "\n\n".join(paragraphs)

        # ------------------------------------------------------------------
        # 표 추출
        # ------------------------------------------------------------------
        tables_md: list[str] = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)

            if rows:
                md_table = rows_to_markdown(rows)
                if md_table:
                    tables_md.append(md_table)

        # ------------------------------------------------------------------
        # 제목 추출
        # ------------------------------------------------------------------
        title = path.stem
        if doc.core_properties.title:
            title = doc.core_properties.title

        # ------------------------------------------------------------------
        # 메타데이터
        # ------------------------------------------------------------------
        metadata: dict[str, object] = {}
        props = doc.core_properties

        if props.author:
            metadata["author"] = props.author
        if props.created:
            metadata["date"] = props.created.strftime("%Y-%m-%d")

        # 파일명에서 날짜 추출 시도 (YYMMDD)
        date_from_name = extract_date_from_filename(path.stem)
        if date_from_name and "date" not in metadata:
            metadata["date"] = date_from_name

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )
