"""다양한 파일 형식의 문서 파서입니다."""

from .base import BaseParser, ParsedDocument, ParserRegistry
from .pdf import PDFParser
from .hwpx import HWPXParser
from .html import HTMLParser
from .text import TextParser

# DOCX 파서는 선택적 종속성입니다
try:
    from .docx import DOCXParser  # noqa: F401
except ImportError:
    DOCXParser = None  # type: ignore

# HWP 바이너리 파서는 선택적 종속성입니다 (olefile 필요)
try:
    from .hwp import HWPParser  # noqa: F401
except ImportError:
    HWPParser = None  # type: ignore

# DOC 바이너리 파서는 선택적 종속성입니다 (olefile 필요)
try:
    from .doc import DOCParser  # noqa: F401
except ImportError:
    DOCParser = None  # type: ignore

# PPTX 파서는 선택적 종속성입니다 (python-pptx 필요)
try:
    from .pptx import PPTXParser  # noqa: F401
except ImportError:
    PPTXParser = None  # type: ignore

# PPT 바이너리 파서는 선택적 종속성입니다 (olefile 필요)
try:
    from .ppt import PPTParser  # noqa: F401
except ImportError:
    PPTParser = None  # type: ignore

# XLSX 파서는 선택적 종속성입니다 (openpyxl 필요)
try:
    from .xlsx import XLSXParser  # noqa: F401
except ImportError:
    XLSXParser = None  # type: ignore

# XLS 파서는 선택적 종속성입니다 (xlrd 필요)
try:
    from .xls import XLSParser  # noqa: F401
except ImportError:
    XLSParser = None  # type: ignore

__all__ = [
    "BaseParser",
    "ParsedDocument",
    "ParserRegistry",
    "PDFParser",
    "HWPXParser",
    "HWPParser",
    "HTMLParser",
    "TextParser",
    "DOCXParser",
    "DOCParser",
    "PPTXParser",
    "PPTParser",
    "XLSXParser",
    "XLSParser",
]

registry = ParserRegistry()
registry.register(PDFParser)
registry.register(HWPXParser)
registry.register(HTMLParser)
registry.register(TextParser)

if DOCXParser is not None:
    registry.register(DOCXParser)

if HWPParser is not None:
    registry.register(HWPParser)

if DOCParser is not None:
    registry.register(DOCParser)

if PPTXParser is not None:
    registry.register(PPTXParser)

if PPTParser is not None:
    registry.register(PPTParser)

if XLSXParser is not None:
    registry.register(XLSXParser)

if XLSParser is not None:
    registry.register(XLSParser)
