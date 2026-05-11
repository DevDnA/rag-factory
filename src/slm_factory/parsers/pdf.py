"""PDF 파서 — pymupdf4llm(LLM 최적화 마크다운 + OCR) 또는 PyMuPDF(fitz) 폴백.

``pymupdf4llm``이 설치되어 있으면 ``to_markdown()``을 사용하여 heading 구조,
표, 볼드/이탤릭, 코드 블록까지 보존된 마크다운을 추출합니다.
스캔 PDF의 경우 OCR을 자동으로 수행합니다 (Tesseract + opencv-python 필요).

``pymupdf4llm``이 설치되지 않았으면 기존 PyMuPDF ``get_text("text")`` +
``find_tables()`` 기반 추출로 폴백합니다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import fitz  # PyMuPDF

from ..utils import get_logger
from .base import (
    BaseParser,
    ParsedDocument,
    extract_date_from_filename,
    rows_to_markdown,
)

logger = get_logger("parsers.pdf")

try:
    import pymupdf4llm

    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False

# 제거할 페이지 번호 라인의 패턴 (예: "- 1 -", "— 3 —", 독립형 숫자)
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*[-—–]\s*\d+\s*[-—–]\s*$", re.MULTILINE),  # - 1 -, — 12 —
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # 독립형 페이지 번호
    re.compile(r"^\s*page\s+\d+\s*$", re.MULTILINE | re.IGNORECASE),  # "Page 3"
]


def _clean_page_numbers(text: str) -> str:
    """추출된 텍스트에서 일반적인 페이지 번호 패턴을 제거합니다."""
    for pattern in _PAGE_NUM_PATTERNS:
        text = pattern.sub("", text)
    return text


def _table_to_markdown(table: fitz.table.Table) -> str:  # type: ignore[name-defined]
    """PyMuPDF Table 객체를 마크다운 표 문자열로 변환합니다."""
    try:
        raw_rows = table.extract()
    except Exception:
        logger.debug("Table extraction failed", exc_info=True)
        return ""

    if not raw_rows:
        return ""

    def _cell(val: object) -> str:
        if val is None:
            return ""
        return str(val).replace("\n", " ").strip()

    table_rows = [[_cell(c) for c in row] for row in raw_rows]
    return rows_to_markdown(table_rows)


def _extract_tables_from_markdown(md_text: str) -> list[str]:
    """마크다운 텍스트에서 표 블록을 추출합니다."""
    tables: list[str] = []
    current_table: list[str] = []
    in_table = False

    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            current_table.append(stripped)
        else:
            if in_table and current_table:
                tables.append("\n".join(current_table))
                current_table = []
            in_table = False

    if current_table:
        tables.append("\n".join(current_table))

    return tables


class PDFParser(BaseParser):
    """PDF 문서를 파싱합니다.

    pymupdf4llm 설치 시 LLM 최적화 마크다운 + OCR을 사용하고,
    미설치 시 기존 PyMuPDF(fitz) 기반으로 폴백합니다.
    """

    extensions: ClassVar[list[str]] = [".pdf"]

    def can_parse_content(self, path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                return f.read(5).startswith(b"%PDF")
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """PDF 파일에서 텍스트와 표를 추출합니다.

        매개변수
        ----------
        path:
            PDF 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            PyMuPDF가 파일을 열 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if HAS_PYMUPDF4LLM:
            return self._parse_with_pymupdf4llm(path)
        return self._parse_with_fitz(path)

    def _parse_with_pymupdf4llm(self, path: Path) -> ParsedDocument:
        """pymupdf4llm을 사용한 LLM 최적화 추출 (OCR 포함)."""
        try:
            md_text = pymupdf4llm.to_markdown(
                str(path),
                use_ocr=True,
                ocr_language="kor+eng",
                show_progress=False,
            )
        except Exception as exc:
            logger.warning(
                "pymupdf4llm 추출 실패, fitz 폴백: %s (%s)",
                path.name,
                exc,
            )
            return self._parse_with_fitz(path)

        content = _clean_page_numbers(md_text).strip()
        content = re.sub(r"\n{3,}", "\n\n", content)

        tables_md = _extract_tables_from_markdown(md_text)

        doc = fitz.open(str(path))  # type: ignore[attr-defined]
        try:
            pdf_meta = doc.metadata or {}
            metadata = self._build_metadata(pdf_meta, len(doc), path)
            title = (pdf_meta.get("title") or "").strip() or path.stem
        finally:
            doc.close()

        logger.info(
            "pymupdf4llm으로 PDF 추출 완료: %s (%d자, 표 %d개)",
            path.name,
            len(content),
            len(tables_md),
        )

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )

    def _parse_with_fitz(self, path: Path) -> ParsedDocument:
        """기존 PyMuPDF(fitz) 기반 추출 (pymupdf4llm 미설치 시 폴백)."""
        try:
            doc = fitz.open(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(
                f"PDF 파일을 열 수 없습니다: {path}\n"
                f"원인: 파일이 손상되었거나, 암호화되었거나, 지원하지 않는 형식일 수 있습니다.\n"
                f"해결: 다른 PDF 뷰어에서 열어 정상 파일인지 확인하세요."
            ) from exc

        try:
            pages_text: list[str] = []
            for page in doc:
                try:
                    text = page.get_text("text")
                    if text:
                        pages_text.append(text)
                except Exception:
                    logger.warning(
                        "Could not extract text from page %d of %s",
                        page.number,
                        path.name,
                    )

            raw_text = "\n\n".join(pages_text)
            content = _clean_page_numbers(raw_text).strip()
            content = re.sub(r"\n{3,}", "\n\n", content)

            tables_md: list[str] = []
            try:
                for page in doc:
                    tab_finder = page.find_tables()
                    for table in tab_finder.tables:
                        md = _table_to_markdown(table)
                        if md:
                            tables_md.append(md)
            except AttributeError:
                logger.debug(
                    "Table extraction unavailable (PyMuPDF version lacks find_tables)"
                )
            except Exception:
                logger.debug("Table extraction failed for %s", path.name, exc_info=True)

            pdf_meta = doc.metadata or {}
            metadata = self._build_metadata(pdf_meta, len(doc), path)
            title = (pdf_meta.get("title") or "").strip() or path.stem
        finally:
            doc.close()

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )

    @staticmethod
    def _build_metadata(
        pdf_meta: dict[str, object],
        page_count: int,
        path: Path,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {}
        if pdf_meta.get("author"):
            metadata["author"] = pdf_meta["author"]
        if pdf_meta.get("creationDate"):
            metadata["creation_date"] = pdf_meta["creationDate"]
        if pdf_meta.get("modDate"):
            metadata["mod_date"] = pdf_meta["modDate"]
        if pdf_meta.get("subject"):
            metadata["subject"] = pdf_meta["subject"]
        metadata["page_count"] = page_count
        date_from_name = extract_date_from_filename(path.stem)
        if date_from_name:
            metadata["date"] = date_from_name
        return metadata
