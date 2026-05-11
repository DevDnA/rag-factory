"""HWPX 파서 — 한국 .hwpx 문서에서 텍스트와 표를 추출합니다.

HWPX 파일은 XML을 포함하는 ZIP 아카이브입니다 (section0.xml, section1.xml, ...).
단락은 <hp:p><hp:t> 태그에, 표는 <hp:tbl><hp:tr><hp:tc> 태그에 있습니다.
멀티섹션 문서의 경우 모든 ``Contents/section*.xml`` 파일을 순서대로 병합합니다.

선택사항: 한국어 띄어쓰기 교정을 위한 kiwipiepy (uv sync --extra korean).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup

from ..utils import get_logger
from .base import (
    BaseParser,
    ParsedDocument,
    extract_date_from_filename,
    rows_to_markdown,
)

logger = get_logger("parsers.hwpx")

try:
    from kiwipiepy import Kiwi

    _kiwi: Kiwi | None = Kiwi()
except ImportError:
    _kiwi = None


def _table_to_markdown(table_element) -> str:
    """hp:tbl BeautifulSoup 요소를 마크다운 표 형식으로 변환합니다."""
    try:
        rows = table_element.find_all("hp:tr", recursive=False)
        if not rows:
            return ""

        # 각 행에서 셀 추출
        table_rows = []
        for row in rows:
            cells = row.find_all("hp:tc", recursive=False)
            row_data = []
            for cell in cells:
                # 셀에서 모든 텍스트 가져오기
                cell_text = cell.get_text(separator=" ", strip=True)
                row_data.append(cell_text)
            if row_data:
                table_rows.append(row_data)

        return rows_to_markdown(table_rows)
    except Exception:
        logger.debug("Failed to convert table to markdown", exc_info=True)
        return ""


class HWPXParser(BaseParser):
    """HWPX (한국 문서) 파일을 파싱합니다."""

    extensions: ClassVar[list[str]] = [".hwpx"]

    def can_parse_content(self, path: Path) -> bool:
        try:
            if not zipfile.is_zipfile(str(path)):
                return False
            with zipfile.ZipFile(path, "r") as zf:
                return any(
                    re.match(r"Contents/section\d+\.xml$", n) for n in zf.namelist()
                )
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """HWPX 파일에서 텍스트와 표를 추출합니다.

        매개변수
        ----------
        path:
            HWPX 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            HWPX 파일을 열거나 파싱할 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HWPX not found: {path}")

        if not zipfile.is_zipfile(str(path)):
            raise RuntimeError(
                f"유효한 HWPX(ZIP) 파일이 아닙니다: {path}\n"
                f"원인: 파일이 손상되었거나 지원하지 않는 형식입니다."
            )

        try:
            with zipfile.ZipFile(path, "r") as zf:
                section_files = sorted(
                    name
                    for name in zf.namelist()
                    if re.match(r"Contents/section\d+\.xml$", name)
                )
                if not section_files:
                    raise FileNotFoundError("No section XML files found")
                xml_parts = [zf.read(name).decode("utf-8") for name in section_files]
                if len(section_files) > 1:
                    logger.info(
                        "HWPX 멀티섹션 감지: %d개 섹션 (%s)",
                        len(section_files),
                        path.name,
                    )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"HWPX 파일 구조가 올바르지 않습니다: {path}\n"
                f"원인: Contents/section*.xml을 찾을 수 없습니다. 파일이 손상되었을 수 있습니다.\n"
                f"해결: 한글에서 다시 저장하거나 PDF로 변환하여 사용하세요."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"HWPX 파일을 읽을 수 없습니다: {path}\n"
                f"원인: ZIP 구조가 손상되었거나 암호화되어 있을 수 있습니다.\n"
                f"해결: 한글에서 열어 '다른 이름으로 저장'하거나 PDF로 변환하세요."
            ) from exc

        # ------------------------------------------------------------------
        # BeautifulSoup으로 각 섹션 XML 파싱 및 병합
        # ------------------------------------------------------------------
        paragraphs: list[str] = []
        tables_md: list[str] = []

        for xml_content in xml_parts:
            try:
                soup = BeautifulSoup(xml_content, "xml")
            except Exception as exc:
                raise RuntimeError(
                    f"HWPX 내부 XML 파싱에 실패했습니다: {path}\n"
                    f"원인: XML 구조가 올바르지 않습니다.\n"
                    f"해결: 한글에서 다시 저장하거나 PDF로 변환하여 사용하세요."
                ) from exc

            # ------------------------------------------------------------------
            # 단락 추출 (중첩된 것 건너뜀)
            # ------------------------------------------------------------------
            for p_tag in soup.find_all("hp:p"):
                # 중첩된 단락 건너뜀
                if p_tag.find_parent("hp:p") is not None:
                    continue

                # hp:t 자식 태그에서 텍스트 추출
                text_parts = []
                for t_tag in p_tag.find_all("hp:t"):
                    text = t_tag.get_text(strip=True)
                    if text:
                        text_parts.append(text)

                if text_parts:
                    para_text = "".join(text_parts)
                    if _kiwi is not None:
                        try:
                            result = _kiwi.space(para_text)
                            para_text = (
                                result if isinstance(result, str) else "".join(result)
                            )
                        except Exception:
                            logger.debug(
                                "kiwipiepy spacing failed, using original text"
                            )
                    paragraphs.append(para_text)

            # ------------------------------------------------------------------
            # 표 추출
            # ------------------------------------------------------------------
            for table in soup.find_all("hp:tbl"):
                md = _table_to_markdown(table)
                if md:
                    tables_md.append(md)

        content = "\n\n".join(paragraphs).strip()

        # 과도한 빈 줄 정규화
        content = re.sub(r"\n{3,}", "\n\n", content)

        # ------------------------------------------------------------------
        # 메타데이터
        # ------------------------------------------------------------------
        metadata: dict[str, object] = {}

        # 파일명에서 날짜 추출 시도 (YYMMDD)
        date_from_name = extract_date_from_filename(path.stem)
        if date_from_name:
            metadata["date"] = date_from_name

        # ------------------------------------------------------------------
        # 제목: 파일명 사용
        # ------------------------------------------------------------------
        title = path.stem

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )
