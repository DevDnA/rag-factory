"""HTML 파서 — BeautifulSoup + lxml을 사용하여 .html/.htm 파일에서 텍스트를 추출합니다."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup

from ..utils import get_logger
from .base import (
    BaseParser,
    ParsedDocument,
    detect_encoding,
    extract_date_from_filename,
    rows_to_markdown,
)

logger = get_logger("parsers.html")


def _table_to_markdown(table_element) -> str:
    """HTML 표 요소를 마크다운 형식으로 변환합니다."""
    try:
        rows = table_element.find_all("tr")
        if not rows:
            return ""

        table_rows = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            row_data = []
            for cell in cells:
                cell_text = cell.get_text(separator=" ", strip=True)
                row_data.append(cell_text)
            if row_data:
                table_rows.append(row_data)

        return rows_to_markdown(table_rows)
    except Exception:
        logger.debug("Failed to convert table to markdown", exc_info=True)
        return ""


class HTMLParser(BaseParser):
    """HTML 문서를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".html", ".htm"]

    def can_parse_content(self, path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                head = f.read(1024).lower().lstrip()
                return head.startswith((b"<!doctype html", b"<html", b"<!doctype"))
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """HTML 파일에서 텍스트와 표를 추출합니다.

        매개변수
        ----------
        path:
            HTML 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            HTML 파일을 파싱할 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")

        # ------------------------------------------------------------------
        # 인코딩 감지로 파일 읽기
        # ------------------------------------------------------------------
        try:
            raw_content = path.read_bytes()
            encoding = detect_encoding(raw_content)
            html_content = raw_content.decode(encoding)
        except Exception as exc:
            raise RuntimeError(
                f"HTML 파일을 읽을 수 없습니다: {path}\n"
                f"원인: 파일 인코딩이 UTF-8이 아니거나 접근 권한이 없을 수 있습니다.\n"
                f"해결: UTF-8로 다시 저장하거나 파일 권한을 확인하세요."
            ) from exc

        # ------------------------------------------------------------------
        # HTML 파싱
        # ------------------------------------------------------------------
        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception as exc:
            raise RuntimeError(
                f"HTML 파싱에 실패했습니다: {path}\n"
                f"원인: 유효하지 않은 HTML 구조이거나 인코딩 문제일 수 있습니다.\n"
                f"해결: 브라우저에서 정상 표시되는지 확인하고, UTF-8로 저장하세요."
            ) from exc

        # ------------------------------------------------------------------
        # 텍스트 추출 전에 script 및 style 태그 제거
        # ------------------------------------------------------------------
        for tag in soup.find_all(["script", "style", "nav"]):
            tag.decompose()

        # ------------------------------------------------------------------
        # 제목 추출
        # ------------------------------------------------------------------
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        if not title:
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text(strip=True)

        if not title:
            title = path.stem

        # ------------------------------------------------------------------
        # heading 태그를 마크다운 형식으로 변환
        # ------------------------------------------------------------------
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                heading_text = heading.get_text(strip=True)
                if heading_text:
                    heading.replace_with(f"\n{'#' * level} {heading_text}\n")

        # ------------------------------------------------------------------
        # 주요 텍스트 추출
        # ------------------------------------------------------------------
        content = soup.get_text(separator="\n", strip=True)

        # 과도한 빈 줄 정규화
        content = re.sub(r"\n{3,}", "\n\n", content)

        # ------------------------------------------------------------------
        # 표 추출
        # ------------------------------------------------------------------
        tables_md: list[str] = []
        for table in soup.find_all("table"):
            md = _table_to_markdown(table)
            if md:
                tables_md.append(md)

        # ------------------------------------------------------------------
        # 메타데이터
        # ------------------------------------------------------------------
        metadata: dict[str, object] = {}

        # 파일명에서 날짜 추출 시도 (YYMMDD)
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
