"""PPT (PowerPoint 97-2003) 바이너리 파서 — olefile로 텍스트를 추출합니다.

PowerPoint Binary Format(.ppt)은 OLE2 복합 문서입니다. ``PowerPoint Document``
스트림의 레코드를 순회하며 TextCharsAtom(유니코드)과 TextBytesAtom(ANSI)에서
슬라이드 텍스트를 추출합니다.

선택적 의존성: ``uv sync --extra hwp`` (olefile)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import ClassVar

from ..models import ParsedDocument
from ..utils import get_logger
from .base import BaseParser, extract_date_from_filename

logger = get_logger("parsers.ppt")

try:
    import olefile

    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False

# PPT 레코드 타입 상수
_RT_TEXT_CHARS_ATOM = 0x0FA0  # 유니코드(UTF-16LE) 텍스트
_RT_TEXT_BYTES_ATOM = 0x0FA8  # ANSI(CP1252) 텍스트
_PPT_STREAM_NAME = "PowerPoint Document"
_RECORD_HEADER_SIZE = 8


def _extract_text_from_ppt(ole: olefile.OleFileIO) -> list[str]:  # type: ignore[name-defined]
    """PowerPoint Document 스트림에서 모든 텍스트를 추출합니다.

    PPT 레코드 구조 (8바이트 헤더):
      - recVer(4bit) + recInstance(12bit): 2바이트
      - recType: 2바이트
      - recLen: 4바이트
    TextCharsAtom / TextBytesAtom 레코드에서 텍스트를 읽습니다.
    """
    data = ole.openstream(_PPT_STREAM_NAME).read()
    texts: list[str] = []
    offset = 0

    while offset + _RECORD_HEADER_SIZE <= len(data):
        rec_type = struct.unpack_from("<H", data, offset + 2)[0]
        rec_len = struct.unpack_from("<I", data, offset + 4)[0]
        payload_start = offset + _RECORD_HEADER_SIZE

        if payload_start + rec_len > len(data):
            break

        if rec_type == _RT_TEXT_CHARS_ATOM and rec_len >= 2:
            payload = data[payload_start : payload_start + rec_len]
            text = payload.decode("utf-16-le", errors="replace").strip()
            if text:
                texts.append(text)

        elif rec_type == _RT_TEXT_BYTES_ATOM and rec_len >= 1:
            payload = data[payload_start : payload_start + rec_len]
            text = payload.decode("cp1252", errors="replace").strip()
            if text:
                texts.append(text)

        # 컨테이너 레코드(recVer == 0xF)는 자식을 포함하므로 내부 순회
        rec_ver = struct.unpack_from("<H", data, offset)[0] & 0x0F
        if rec_ver == 0x0F:
            offset += _RECORD_HEADER_SIZE
        else:
            offset += _RECORD_HEADER_SIZE + rec_len

    return texts


class PPTParser(BaseParser):
    """PPT (PowerPoint 97-2003) 바이너리 프레젠테이션을 파싱합니다."""

    extensions: ClassVar[list[str]] = [".ppt"]

    def can_parse_content(self, path: Path) -> bool:
        if not HAS_OLEFILE:
            return False
        try:
            if not olefile.isOleFile(str(path)):
                return False
            with olefile.OleFileIO(str(path)) as ole:
                return ole.exists(_PPT_STREAM_NAME)
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """PPT 파일에서 텍스트를 추출합니다.

        매개변수
        ----------
        path:
            PPT 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용과 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            PPT 파일을 파싱할 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PPT file not found: {path}")

        if not HAS_OLEFILE:
            raise RuntimeError(
                f"PPT 파일을 파싱하려면 olefile이 필요합니다: {path}\n"
                f"설치: uv sync --extra hwp"
            )

        if not olefile.isOleFile(str(path)):
            raise RuntimeError(
                f"유효한 PPT(OLE2) 파일이 아닙니다: {path}\n"
                f"원인: 파일이 손상되었거나 지원하지 않는 형식입니다."
            )

        try:
            ole = olefile.OleFileIO(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"PPT 파일을 열 수 없습니다: {path}\n"
                f"원인: OLE2 구조가 손상되었거나 암호화되어 있을 수 있습니다.\n"
                f"해결: PowerPoint에서 .pptx로 다시 저장하세요."
            ) from exc

        try:
            texts = _extract_text_from_ppt(ole)
            content = "\n\n".join(texts)

            metadata: dict[str, object] = {}
            date_from_name = extract_date_from_filename(path.stem)
            if date_from_name:
                metadata["date"] = date_from_name

            title = path.stem
            try:
                meta = ole.get_metadata()
                if hasattr(meta, "author") and meta.author:
                    metadata["author"] = meta.author
                if hasattr(meta, "title") and meta.title:
                    title = meta.title
                if hasattr(meta, "create_time") and meta.create_time:
                    metadata.setdefault("date", meta.create_time.strftime("%Y-%m-%d"))
            except Exception:
                logger.debug("PPT 메타데이터 추출 실패", exc_info=True)
        finally:
            ole.close()

        if not content.strip():
            logger.warning("PPT 파일에서 텍스트를 추출하지 못했습니다: %s", path.name)

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content.strip(),
            tables=[],
            metadata=metadata,
        )
