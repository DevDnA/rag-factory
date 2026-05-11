"""DOC (Word 97-2003) 바이너리 파서 — olefile로 텍스트를 추출합니다.

Word Binary Format(.doc)은 OLE2 복합 문서입니다. ``WordDocument`` 스트림의
FIB(File Information Block)에서 텍스트 위치를 파악하고, Table 스트림의
CLX(Piece Table)를 통해 ANSI/유니코드 텍스트를 추출합니다.

선택적 의존성: ``uv sync --extra hwp`` (olefile)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import ClassVar

from ..models import ParsedDocument
from ..utils import get_logger
from .base import BaseParser, extract_date_from_filename

logger = get_logger("parsers.doc")

try:
    import olefile

    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False

# Word Binary Format 상수
_WORD_MAGIC = 0xA5EC
_FIB_FLAGS_OFFSET = 0x000A
_FIB_FCCLX_OFFSET = 0x01A2
_FIB_LCBCLX_OFFSET = 0x01A6
# Piece descriptor: fCompressed 비트는 FC의 bit 30
_FC_COMPRESSED_BIT = 1 << 30


def _extract_text_from_doc(ole: olefile.OleFileIO) -> str:  # type: ignore[name-defined]
    """Word Binary Format OLE 파일에서 본문 텍스트를 추출합니다.

    FIB → Table Stream → CLX(Piece Table) → WordDocument 텍스트 순서로 파싱합니다.
    """
    word_doc = ole.openstream("WordDocument").read()

    if len(word_doc) < _FIB_LCBCLX_OFFSET + 4:
        return _fallback_text_extraction(ole)

    magic = struct.unpack_from("<H", word_doc, 0)[0]
    if magic != _WORD_MAGIC:
        return _fallback_text_extraction(ole)

    flags = struct.unpack_from("<H", word_doc, _FIB_FLAGS_OFFSET)[0]
    table_name = "1Table" if (flags >> 9) & 1 else "0Table"

    if not ole.exists(table_name):
        return _fallback_text_extraction(ole)

    table_data = ole.openstream(table_name).read()

    fc_clx = struct.unpack_from("<I", word_doc, _FIB_FCCLX_OFFSET)[0]
    lcb_clx = struct.unpack_from("<I", word_doc, _FIB_LCBCLX_OFFSET)[0]

    if lcb_clx == 0 or fc_clx + lcb_clx > len(table_data):
        return _fallback_text_extraction(ole)

    clx = table_data[fc_clx : fc_clx + lcb_clx]
    return _parse_piece_table(clx, word_doc)


def _parse_piece_table(clx: bytes, word_doc: bytes) -> str:
    """CLX 구조에서 Piece Table(Pcdt)을 찾아 텍스트를 추출합니다.

    CLX = Grpprl*(type 0x01) + Pcdt(type 0x02)
    Pcdt = type(1) + size(4) + PLC(piece descriptors)
    PLC = CP[n+1] + PCD[n], PCD = 8 bytes each
    """
    offset = 0
    text_parts: list[str] = []

    while offset < len(clx):
        clx_type = clx[offset]
        offset += 1

        if clx_type == 0x01:
            if offset + 2 > len(clx):
                break
            grpprl_size = struct.unpack_from("<H", clx, offset)[0]
            offset += 2 + grpprl_size
            continue

        if clx_type == 0x02:
            if offset + 4 > len(clx):
                break
            pcdt_size = struct.unpack_from("<I", clx, offset)[0]
            offset += 4
            plc_data = clx[offset : offset + pcdt_size]
            text_parts = _extract_from_plc(plc_data, word_doc)
            break

        break

    return "\n".join(text_parts)


def _extract_from_plc(plc: bytes, word_doc: bytes) -> list[str]:
    """PLC(Piecetable)에서 각 piece의 텍스트를 추출합니다.

    PLC 구조: CP[0..n] (각 4바이트) + PCD[0..n-1] (각 8바이트)
    PCD 내부: FC(4바이트, bit 30 = fCompressed) + prm(4바이트)
    """
    pcd_size = 8
    # n+1 개의 CP + n 개의 PCD = plc 길이
    # (n+1)*4 + n*8 = len(plc) → n = (len(plc) - 4) / 12
    if len(plc) < 16:
        return []

    n_pieces = (len(plc) - 4) // 12
    if n_pieces <= 0:
        return []

    cp_array_end = (n_pieces + 1) * 4
    if cp_array_end + n_pieces * pcd_size > len(plc):
        return []

    texts: list[str] = []
    for i in range(n_pieces):
        cp_start = struct.unpack_from("<I", plc, i * 4)[0]
        cp_end = struct.unpack_from("<I", plc, (i + 1) * 4)[0]
        char_count = cp_end - cp_start

        if char_count <= 0:
            continue

        pcd_offset = cp_array_end + i * pcd_size
        # PCD: 2 bytes (unused) + 4 bytes (fc) + 2 bytes (prm)
        fc_raw = struct.unpack_from("<I", plc, pcd_offset + 2)[0]
        is_compressed = bool(fc_raw & _FC_COMPRESSED_BIT)
        fc = fc_raw & ~_FC_COMPRESSED_BIT

        if is_compressed:
            byte_offset = fc // 2
            byte_len = char_count
            if byte_offset + byte_len > len(word_doc):
                continue
            raw = word_doc[byte_offset : byte_offset + byte_len]
            try:
                texts.append(raw.decode("cp1252", errors="replace"))
            except Exception:
                texts.append(raw.decode("latin-1", errors="replace"))
        else:
            byte_offset = fc
            byte_len = char_count * 2
            if byte_offset + byte_len > len(word_doc):
                continue
            raw = word_doc[byte_offset : byte_offset + byte_len]
            texts.append(raw.decode("utf-16-le", errors="replace"))

    return texts


def _fallback_text_extraction(ole: olefile.OleFileIO) -> str:  # type: ignore[name-defined]
    """Piece Table 파싱 실패 시 WordDocument 스트림에서 직접 텍스트를 추출합니다.

    정규 파싱이 불가능한 손상/비표준 파일에 대한 최후 수단입니다.
    """
    try:
        data = ole.openstream("WordDocument").read()
    except Exception:
        return ""

    # UTF-16LE로 디코딩 시도 (제어 문자 필터링)
    try:
        text = data.decode("utf-16-le", errors="ignore")
    except Exception:
        text = data.decode("cp1252", errors="ignore")

    lines: list[str] = []
    for line in text.split("\n"):
        cleaned = "".join(ch for ch in line if ch.isprintable() or ch in "\t\n")
        cleaned = cleaned.strip()
        if len(cleaned) > 3:
            lines.append(cleaned)

    return "\n".join(lines)


class DOCParser(BaseParser):
    """DOC (Word 97-2003) 바이너리 문서를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".doc"]

    def can_parse_content(self, path: Path) -> bool:
        if not HAS_OLEFILE:
            return False
        try:
            if not olefile.isOleFile(str(path)):
                return False
            with olefile.OleFileIO(str(path)) as ole:
                return ole.exists("WordDocument")
        except Exception:
            return False

    def parse(self, path: Path) -> ParsedDocument:
        """DOC 파일에서 텍스트를 추출합니다.

        매개변수
        ----------
        path:
            DOC 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용과 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            DOC 파일을 파싱할 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DOC file not found: {path}")

        if not HAS_OLEFILE:
            raise RuntimeError(
                f"DOC 파일을 파싱하려면 olefile이 필요합니다: {path}\n"
                f"설치: uv sync --extra hwp"
            )

        if not olefile.isOleFile(str(path)):
            raise RuntimeError(
                f"유효한 DOC(OLE2) 파일이 아닙니다: {path}\n"
                f"원인: 파일이 손상되었거나 지원하지 않는 형식입니다."
            )

        try:
            ole = olefile.OleFileIO(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"DOC 파일을 열 수 없습니다: {path}\n"
                f"원인: OLE2 구조가 손상되었거나 암호화되어 있을 수 있습니다.\n"
                f"해결: Word에서 .docx로 다시 저장하세요."
            ) from exc

        try:
            content = _extract_text_from_doc(ole)

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
                logger.debug("DOC 메타데이터 추출 실패", exc_info=True)
        finally:
            ole.close()

        if not content.strip():
            logger.warning("DOC 파일에서 텍스트를 추출하지 못했습니다: %s", path.name)

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content.strip(),
            tables=[],
            metadata=metadata,
        )
