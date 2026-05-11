"""HWP 바이너리 파서 — 한글 .hwp(v5) 문서에서 텍스트를 추출합니다.

HWP5 파일은 OLE2 복합 문서입니다. ``olefile``로 내부 스트림에 접근하고,
``pyhwp``가 설치되어 있으면 ``hwp5`` 라이브러리를 사용하여 텍스트를
구조적으로 추출합니다. 미설치 시 ``olefile`` 기반 원시 텍스트 추출로 폴백합니다.

선택적 의존성: ``uv sync --extra hwp``
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path
from typing import ClassVar

from ..utils import get_logger
from .base import BaseParser, ParsedDocument, extract_date_from_filename

logger = get_logger("parsers.hwp")

_CJK = r"\u3400-\u4DBF\u4E00-\u9FFF"
_HWP_ARTIFACT_RE = re.compile(
    rf"[{_CJK}]\n[\u0080-\u024F]\n"
    rf"|[{_CJK}][\u0800-\u09FF]"
    r"|[\u0200-\u02FF]"
    rf"|(?<=[\uAC00-\uD7A3\d\s])[{_CJK}](?=[\n\s\uAC00-\uD7A3])"
    rf"|(?<=[\uAC00-\uD7A3])[{_CJK}](?=$)"
    r"|[\u2C00-\u2C5F]"
    r"|[\U000F0000-\U000FFFFF]"
)


def _clean_hwp_text(text: str) -> str:
    """HWP 파서 출력에서 제어 문자 아티팩트를 제거합니다.

    HWP5 바이너리에서 추출된 텍스트에는 섹션 마커, 서식 바이트 등이
    한자(CJK)나 라틴 확장 문자로 디코딩되어 남습니다.
    패턴: "주요현황艰\\nȃ\\n", "2,476浵ࡦ" 등
    """
    cleaned = _HWP_ARTIFACT_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


try:
    import olefile

    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False


class HWPParser(BaseParser):
    """HWP5 (바이너리 한글) 문서를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".hwp"]

    def can_parse_content(self, path: Path) -> bool:
        if not HAS_OLEFILE:
            return False
        try:
            return olefile.isOleFile(str(path))
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """HWP 파일에서 텍스트를 추출합니다.

        매개변수
        ----------
        path:
            HWP 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용과 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            HWP 파일을 파싱할 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HWP not found: {path}")

        if not HAS_OLEFILE:
            raise RuntimeError(
                f"HWP 파일을 파싱하려면 olefile이 필요합니다: {path}\n"
                f"설치: uv sync --extra hwp"
            )

        if not olefile.isOleFile(str(path)):
            raise RuntimeError(
                f"유효한 HWP(OLE2) 파일이 아닙니다: {path}\n"
                f"원인: 파일이 손상되었거나 지원하지 않는 형식입니다."
            )

        try:
            ole = olefile.OleFileIO(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"HWP 파일을 열 수 없습니다: {path}\n"
                f"원인: OLE2 구조가 손상되었거나 암호화되어 있을 수 있습니다.\n"
                f"해결: 한글에서 열어 '다른 이름으로 저장'하거나 PDF로 변환하세요."
            ) from exc

        try:
            header_data = ole.openstream("FileHeader").read()
            is_compressed = (
                bool(header_data[36] & 1) if len(header_data) > 36 else False
            )

            paragraphs = self._extract_text_from_sections(ole, is_compressed)

            content = "\n\n".join(paragraphs).strip()
            content = _clean_hwp_text(content)

            metadata: dict[str, object] = {}
            date_from_name = extract_date_from_filename(path.stem)
            if date_from_name:
                metadata["date"] = date_from_name

            if ole.exists("\x05HwpSummaryInformation"):
                try:
                    meta = ole.get_metadata()
                    if meta.author:
                        metadata["author"] = meta.author
                    if meta.title:
                        metadata["title_meta"] = meta.title
                except Exception:
                    logger.debug("HWP 메타데이터 추출 실패", exc_info=True)

            title = str(metadata.get("title_meta", "")) or path.stem
        finally:
            ole.close()

        if not content:
            logger.warning("HWP 파일에서 텍스트를 추출하지 못했습니다: %s", path.name)

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=[],
            metadata=metadata,
        )

    @staticmethod
    def _extract_text_from_sections(
        ole: olefile.OleFileIO,  # type: ignore[name-defined]
        is_compressed: bool,
    ) -> list[str]:
        """BodyText/Section* 스트림에서 텍스트를 추출합니다."""
        paragraphs: list[str] = []

        section_streams = sorted(
            name
            for name in ole.listdir()
            if len(name) == 2
            and name[0] == "BodyText"
            and name[1].startswith("Section")
        )

        if not section_streams:
            logger.warning("BodyText/Section* 스트림을 찾을 수 없습니다")
            return paragraphs

        for stream_name in section_streams:
            try:
                data = ole.openstream(stream_name).read()
                if is_compressed:
                    data = zlib.decompress(data, -15)

                text = _extract_text_from_bodytext(data)
                if text:
                    paragraphs.extend(text)
            except Exception:
                logger.debug(
                    "섹션 스트림 처리 실패: %s",
                    "/".join(stream_name),
                    exc_info=True,
                )

        return paragraphs


def _extract_text_from_bodytext(data: bytes) -> list[str]:
    """HWP5 BodyText 바이너리 데이터에서 텍스트 문자열을 추출합니다.

    HWP5 레코드 구조: 4바이트 헤더 (tag_id[10bit] + level[10bit] + size[12bit])
    HWPTAG_PARA_TEXT (tag_id=67)의 payload에서 UTF-16LE 텍스트를 읽습니다.
    """
    HWPTAG_PARA_TEXT = 67
    paragraphs: list[str] = []
    offset = 0

    while offset + 4 <= len(data):
        header = struct.unpack_from("<I", data, offset)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        offset += 4

        if size == 0xFFF:
            if offset + 4 > len(data):
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        if offset + size > len(data):
            break

        if tag_id == HWPTAG_PARA_TEXT:
            payload = data[offset : offset + size]
            text = _decode_para_text(payload)
            if text.strip():
                paragraphs.append(text.strip())

        offset += size

    return paragraphs


def _decode_para_text(payload: bytes) -> str:
    """HWPTAG_PARA_TEXT 페이로드에서 텍스트를 디코딩합니다.

    HWP5는 UTF-16LE로 텍스트를 저장하되, 특수 제어 코드(0~31 범위)를
    인라인으로 포함합니다. 제어 코드는 건너뛰고 일반 텍스트만 추출합니다.
    """
    chars: list[str] = []
    i = 0

    while i + 1 < len(payload):
        code = struct.unpack_from("<H", payload, i)[0]

        if code < 32:
            if code in (0, 10, 13):
                if chars and chars[-1] != "\n":
                    chars.append("\n")
                i += 2
            elif code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                i += 16
            else:
                i += 2
        elif 0xD800 <= code <= 0xDBFF:
            if i + 3 < len(payload):
                low = struct.unpack_from("<H", payload, i + 2)[0]
                if 0xDC00 <= low <= 0xDFFF:
                    codepoint = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    chars.append(chr(codepoint))
                    i += 4
                    continue
            i += 2
        else:
            chars.append(chr(code))
            i += 2

    return "".join(chars)
