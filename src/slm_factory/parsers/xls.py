"""XLS (레거시 Excel) 문서 파서입니다.

선택적 의존성: ``uv sync --extra xls``
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..models import ParsedDocument
from ..utils import get_logger
from .base import BaseParser, extract_date_from_filename, rows_to_markdown

logger = get_logger("parsers.xls")


class XLSParser(BaseParser):
    """XLS (Excel 97-2003) 스프레드시트를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".xls"]

    def can_parse_content(self, path: Path) -> bool:
        """XLS 파일의 OLE2 magic bytes를 확인합니다."""
        try:
            with open(path, "rb") as f:
                header = f.read(8)
            # OLE2 Compound Document magic: D0 CF 11 E0 A1 B1 1A E1
            return header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """XLS 파일을 파싱하여 ParsedDocument를 반환합니다.

        각 시트를 마크다운 표로 변환합니다.

        매개변수
        ----------
        path:
            XLS 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        ImportError
            xlrd가 설치되지 않았으면 발생합니다.
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            XLS 파일을 파싱할 수 없으면 발생합니다.
        """
        try:
            import xlrd
        except ImportError as exc:
            raise ImportError(
                "xlrd가 설치되지 않았습니다. uv sync --extra xls 로 설치하세요."
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"XLS file not found: {path}")

        try:
            wb = xlrd.open_workbook(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"XLS 파일을 파싱할 수 없습니다: {path}\n"
                f"원인: 파일이 손상되었거나 xlrd가 지원하지 않는 형식일 수 있습니다.\n"
                f"해결: Excel에서 .xlsx로 다시 저장하세요."
            ) from exc

        # ------------------------------------------------------------------
        # 시트별 텍스트 및 표 추출
        # ------------------------------------------------------------------
        sheet_texts: list[str] = []
        tables_md: list[str] = []

        for sheet_idx in range(wb.nsheets):
            sheet = wb.sheet_by_index(sheet_idx)
            rows: list[list[str]] = []

            for row_idx in range(sheet.nrows):
                cells: list[str] = []
                for col_idx in range(sheet.ncols):
                    cell = sheet.cell(row_idx, col_idx)
                    value = cell.value
                    if isinstance(value, float) and value == int(value):
                        cells.append(str(int(value)))
                    elif value is None or value == "":
                        cells.append("")
                    else:
                        cells.append(str(value).strip())

                # 완전히 빈 행 건너뛰기
                if any(c for c in cells):
                    rows.append(cells)

            if not rows:
                continue

            md_table = rows_to_markdown(rows)
            if md_table:
                header = f"## {sheet.name}" if wb.nsheets > 1 else ""
                if header:
                    sheet_texts.append(f"{header}\n\n{md_table}")
                else:
                    sheet_texts.append(md_table)
                tables_md.append(md_table)

        content = "\n\n".join(sheet_texts)

        # ------------------------------------------------------------------
        # 메타데이터
        # ------------------------------------------------------------------
        title = path.stem
        metadata: dict[str, object] = {
            "sheet_count": wb.nsheets,
        }

        date_from_name = extract_date_from_filename(path.stem)
        if date_from_name:
            metadata["date"] = date_from_name

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )
