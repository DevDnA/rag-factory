"""일반 텍스트 파서 — .txt 및 .md 파일을 직접 읽습니다."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..utils import get_logger
from .base import BaseParser, ParsedDocument, detect_encoding, extract_date_from_filename

logger = get_logger("parsers.text")


class TextParser(BaseParser):
    """일반 텍스트 및 마크다운 파일을 파싱합니다."""

    extensions: ClassVar[list[str]] = [".txt", ".md"]

    def parse(self, path: Path) -> ParsedDocument:
        """일반 텍스트 또는 마크다운 파일에서 텍스트를 추출합니다.

        매개변수
        ----------
        path:
            텍스트 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용 및 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            파일을 읽을 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        # ------------------------------------------------------------------
        # 인코딩 감지로 파일 읽기
        # ------------------------------------------------------------------
        try:
            raw_content = path.read_bytes()
            encoding = detect_encoding(raw_content)
            content = raw_content.decode(encoding)
        except Exception as exc:
            raise RuntimeError(
                f"텍스트 파일을 읽을 수 없습니다: {path}\n"
                f"원인: 파일 인코딩이 UTF-8이 아닐 수 있습니다.\n"
                f"해결: UTF-8로 다시 저장하세요. (VS Code: 하단 인코딩 클릭 → Save with Encoding → UTF-8)"
            ) from exc

        # ------------------------------------------------------------------
        # 제목 추출
        # ------------------------------------------------------------------
        title = ""

        # 마크다운 파일의 경우, 첫 번째 # 제목에서 제목 추출 시도
        if path.suffix.lower() == ".md":
            lines = content.split("\n")
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

        # 파일명으로 폴백
        if not title:
            title = path.stem

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
            tables=[],
            metadata=metadata,
        )
