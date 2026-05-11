#!/usr/bin/env python3
"""
docs/build.py — Markdown → HTML converter for slm-factory documentation.

Converts the 7 .md files in docs/ into .html files matching the existing
hand-crafted template. Uses ONLY Python stdlib (no external dependencies).

Usage:
    python docs/build.py              # Build all pages
    python docs/build.py --extract    # Extract SVGs from existing HTML (one-time)
"""

from __future__ import annotations

import html
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent

PAGES: dict[str, dict] = {
    "guide": {
        "title": "사용 가이드",
        "badge": "Guide",
        "subtitle": "slm-factory 설치부터 모델 배포까지, 단계별로 안내합니다.",
    },
    "cli-reference": {
        "title": "CLI 명령어 레퍼런스",
        "badge": "Reference",
        "subtitle": "slm-factory의 모든 명령어와 옵션을 정리한 공식 레퍼런스입니다.",
    },
    "configuration": {
        "title": "설정 레퍼런스",
        "badge": "Reference",
        "subtitle": "project.yaml의 모든 설정 옵션을 상세히 설명합니다.",
    },
    "architecture": {
        "title": "아키텍처 가이드",
        "badge": "Architecture",
        "subtitle": "slm-factory의 내부 구조와 설계 원칙을 설명합니다.",
    },
    "development": {
        "title": "개발 가이드",
        "badge": "Developer",
        "subtitle": "모듈 구조, API 레퍼런스, 확장 가이드",
    },
    "quick-reference": {
        "title": "빠른 참조",
        "badge": "Quick Ref",
        "subtitle": "자주 사용하는 명령어, 워크플로우, 설정 패턴을 한 페이지로 요약합니다.",
    },
    "integration-guide": {
        "title": "RAG 서비스 가이드",
        "badge": "Integration",
        "subtitle": "SLM + RAG 서비스 구축 방법과 프로덕션 배포 가이드",
    },
}

SIDEBAR_ITEMS = [
    ("section", "시작하기"),
    ("link", "index.html", "문서 홈"),
    ("link", "guide.html", "사용 가이드"),
    ("link", "integration-guide.html", "RAG 서비스 가이드"),
    ("link", "quick-reference.html", "빠른 참조"),
    ("section", "레퍼런스"),
    ("link", "cli-reference.html", "CLI 레퍼런스"),
    ("link", "configuration.html", "설정 레퍼런스"),
    ("section", "심화"),
    ("link", "architecture.html", "아키텍처 가이드"),
    ("link", "development.html", "개발 가이드"),
]


# ---------------------------------------------------------------------------
# SVG Diagram Loader
# ---------------------------------------------------------------------------


def load_diagrams() -> dict[str, str]:
    """Load all SVG diagrams from docs/diagrams/ directory."""
    diagrams: dict[str, str] = {}
    diagrams_dir = DOCS_DIR / "diagrams"
    if not diagrams_dir.exists():
        return diagrams
    for svg_file in diagrams_dir.glob("*.svg"):
        content = svg_file.read_text(encoding="utf-8")
        # Strip XML declaration if present — we inline directly
        content = re.sub(r"<\?xml[^?]*\?>\s*", "", content).strip()
        diagram_id = svg_file.stem  # e.g. "guide-diagram-project"
        diagrams[diagram_id] = content
    return diagrams


# ---------------------------------------------------------------------------
# Markdown → HTML Converter
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Create a URL-safe slug from heading text."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove inline code backticks
    text = text.replace("`", "")
    text = text.strip().lower()
    # Replace spaces/special chars with hyphens
    text = re.sub(r"[^\w가-힣\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def escape(text: str) -> str:
    """HTML-escape text, preserving already-escaped entities."""
    return html.escape(text, quote=False)


def process_inline(text: str) -> str:
    """Process inline markdown formatting: bold, italic, code, links, images."""
    # Inline code (must be first to avoid processing inside code)
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"`([^`]+)`", text):
        parts.append(_process_inline_no_code(text[last : m.start()]))
        parts.append(f"<code>{escape(m.group(1))}</code>")
        last = m.end()
    parts.append(_process_inline_no_code(text[last:]))
    return "".join(parts)


def _process_inline_no_code(text: str) -> str:
    """Process inline formatting excluding code spans."""
    # Images: ![alt](src)
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        r'<img src="\2" alt="\1">',
        text,
    )

    # Links: [text](url)
    def _link_replace(m: re.Match) -> str:
        link_text = m.group(1)
        url = m.group(2)
        # Convert .md links to .html for internal links
        if url.endswith(".md"):
            url = url[:-3] + ".html"
        elif ".md#" in url:
            url = url.replace(".md#", ".html#")
        return f'<a href="{url}">{link_text}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link_replace, text)
    # Bold+italic: ***text*** or ___text___
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Italic: *text* or _text_ (but not inside words for _)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    # HTML entities that markdown uses
    text = text.replace(" -- ", " &mdash; ")
    text = text.replace("-->", "&rarr;")
    text = text.replace("<--", "&larr;")
    text = text.replace(" → ", " &rarr; ")
    text = text.replace(" ← ", " &larr; ")
    return text


class MarkdownConverter:
    """Converts Markdown text to HTML, collecting headings for TOC."""

    def __init__(self, diagrams: dict[str, str] | None = None):
        self.headings: list[tuple[int, str, str]] = []  # (level, id, text)
        self.diagrams = diagrams or {}

    def convert(self, md_text: str) -> str:
        """Convert full markdown text to HTML body content."""
        lines = md_text.split("\n")
        out: list[str] = []
        i = 0
        # Skip the first H1 heading and optional subtitle/blockquote/hr
        # (these are used in the page hero, not body content)
        i = self._skip_frontmatter(lines, i)

        while i < len(lines):
            line = lines[i]

            # Diagram marker: <!-- diagram: xxx -->
            dm = re.match(r"^\s*<!--\s*diagram:\s*(\S+)\s*-->\s*$", line)
            if dm:
                diagram_id = dm.group(1)
                if diagram_id in self.diagrams:
                    out.append('<div class="diagram">')
                    out.append(self.diagrams[diagram_id])
                    out.append("</div>")
                i += 1
                continue

            # HTML comment (pass through)
            if line.strip().startswith("<!--"):
                out.append(line)
                i += 1
                continue

            # Fenced code block
            fence_m = re.match(r"^(`{3,}|~{3,})(\w*)", line)
            if fence_m:
                i, block_html = self._parse_code_block(lines, i, fence_m)
                out.append(block_html)
                continue

            # Horizontal rule
            if re.match(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$", line.strip()):
                out.append("<hr>")
                i += 1
                continue

            # Heading
            hm = re.match(r"^(#{1,6})\s+(.+)$", line)
            if hm:
                level = len(hm.group(1))
                text_raw = hm.group(2).strip()
                text_html = process_inline(text_raw)
                hid = slugify(text_raw)
                self.headings.append((level, hid, text_raw))
                if level == 2:
                    out.append(f'<section id="{hid}">')
                    out.append(f'<h{level} id="{hid}">{text_html}</h{level}>')
                else:
                    out.append(f'<h{level} id="{hid}">{text_html}</h{level}>')
                i += 1
                continue

            # Table
            if (
                "|" in line
                and i + 1 < len(lines)
                and re.match(r"^\s*\|?\s*[-:]+[-|:\s]+$", lines[i + 1])
            ):
                i, table_html = self._parse_table(lines, i)
                out.append(table_html)
                continue

            # Blockquote
            if line.startswith(">"):
                i, bq_html = self._parse_blockquote(lines, i)
                out.append(bq_html)
                continue

            # Unordered list
            if re.match(r"^(\s*)([-*+])\s", line):
                i, list_html = self._parse_list(lines, i, ordered=False)
                out.append(list_html)
                continue

            # Ordered list (including "1b." pattern)
            if re.match(r"^(\s*)\d+[b]?\.\s", line):
                i, list_html = self._parse_list(lines, i, ordered=True)
                out.append(list_html)
                continue

            # Blank line
            if not line.strip():
                i += 1
                continue

            # Paragraph
            i, para_html = self._parse_paragraph(lines, i)
            out.append(para_html)

        # Close any open sections
        result = "\n".join(out)
        result = self._close_sections(result)
        return result

    def _skip_frontmatter(self, lines: list[str], i: int) -> int:
        """Skip the H1, optional blockquote subtitle, optional HR, and TOC."""
        # Skip H1
        if i < len(lines) and lines[i].startswith("# "):
            i += 1
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Skip subtitle (line that doesn't start with # or ---)
        # Could be a blockquote (single or multi-line) or plain text
        if i < len(lines) and lines[i].startswith(">"):
            # Skip ALL consecutive blockquote lines
            while i < len(lines) and lines[i].startswith(">"):
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
        elif (
            i < len(lines)
            and not lines[i].startswith("#")
            and not re.match(r"^\s*---", lines[i])
        ):
            # Plain text subtitle
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
        # Skip HR
        if i < len(lines) and re.match(r"^\s*---", lines[i]):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
        # Skip TOC section (## 목차 + list of links)
        if i < len(lines) and re.match(r"^##\s+목차", lines[i]):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and (
                lines[i].startswith("- ") or not lines[i].strip()
            ):
                i += 1
        # Skip HR after TOC
        if i < len(lines) and re.match(r"^\s*---", lines[i]):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
        return i

    def _parse_code_block(
        self, lines: list[str], i: int, fence_m: re.Match
    ) -> tuple[int, str]:
        """Parse a fenced code block and return (next_index, html)."""
        fence_char = fence_m.group(1)[0]
        fence_len = len(fence_m.group(1))
        lang = fence_m.group(2) or ""
        i += 1
        code_lines: list[str] = []
        while i < len(lines):
            close_m = re.match(
                rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$", lines[i]
            )
            if close_m:
                i += 1
                break
            code_lines.append(lines[i])
            i += 1
        code_text = escape("\n".join(code_lines))
        if lang:
            return i, f'<pre data-lang="{lang}"><code>{code_text}</code></pre>'
        return i, f"<pre><code>{code_text}</code></pre>"

    def _parse_table(self, lines: list[str], i: int) -> tuple[int, str]:
        """Parse a markdown table and return (next_index, html)."""
        # Header row
        header_cells = self._split_table_row(lines[i])
        i += 1
        # Separator row — extract alignments
        sep_cells = self._split_table_row(lines[i])
        aligns: list[str] = []
        for cell in sep_cells:
            cell = cell.strip()
            if cell.startswith(":") and cell.endswith(":"):
                aligns.append("center")
            elif cell.endswith(":"):
                aligns.append("right")
            else:
                aligns.append("")
        i += 1

        rows: list[list[str]] = []
        while i < len(lines) and "|" in lines[i] and lines[i].strip():
            rows.append(self._split_table_row(lines[i]))
            i += 1

        # Build HTML
        parts = ['<div class="table-wrapper">', "<table>", "<thead>", "<tr>"]
        for ci, cell in enumerate(header_cells):
            align = aligns[ci] if ci < len(aligns) else ""
            style = f' style="text-align:{align}"' if align else ""
            parts.append(f"<th{style}>{process_inline(cell.strip())}</th>")
        parts.append("</tr>")
        parts.append("</thead>")
        parts.append("<tbody>")
        for row in rows:
            parts.append("<tr>")
            for ci, cell in enumerate(row):
                align = aligns[ci] if ci < len(aligns) else ""
                style = f' style="text-align:{align}"' if align else ""
                parts.append(f"<td{style}>{process_inline(cell.strip())}</td>")
            parts.append("</tr>")
        parts.append("</tbody>")
        parts.append("</table>")
        parts.append("</div>")
        return i, "\n".join(parts)

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        """Split a table row into cells, handling leading/trailing pipes.

        Escaped pipes (``\\|``) inside cells are preserved as literal ``|``.
        """
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        # Replace escaped pipes with placeholder, split, then restore
        placeholder = "\x00PIPE\x00"
        line = line.replace("\\|", placeholder)
        cells = line.split("|")
        return [cell.replace(placeholder, "|") for cell in cells]

    def _parse_blockquote(self, lines: list[str], i: int) -> tuple[int, str]:
        """Parse a blockquote block and return (next_index, html)."""
        bq_lines: list[str] = []
        while i < len(lines) and (
            lines[i].startswith(">")
            or (
                lines[i].strip()
                and bq_lines
                and not lines[i].startswith("#")
                and not lines[i].startswith("```")
                and not re.match(r"^\s*[-*+]\s", lines[i])
                and not re.match(r"^\s*\d+\.\s", lines[i])
            )
        ):
            text = lines[i]
            if text.startswith("> "):
                text = text[2:]
            elif text.startswith(">"):
                text = text[1:]
            bq_lines.append(text)
            i += 1
        content = process_inline(" ".join(bq_lines))
        return i, f"<blockquote><p>{content}</p></blockquote>"

    def _parse_list(
        self, lines: list[str], i: int, *, ordered: bool
    ) -> tuple[int, str]:
        """Parse a list (ordered or unordered) with nesting support."""
        tag = "ol" if ordered else "ul"
        items: list[str] = []
        base_indent = len(lines[i]) - len(lines[i].lstrip())

        while i < len(lines):
            line = lines[i]
            if not line.strip():
                # Blank line — check if list continues
                if i + 1 < len(lines) and self._is_list_item(
                    lines[i + 1], ordered, base_indent
                ):
                    i += 1
                    continue
                break

            indent = len(line) - len(line.lstrip())

            if indent > base_indent and items:
                # Nested content — could be a nested list or continuation
                if self._is_list_item(line, not ordered, indent) or self._is_list_item(
                    line, ordered, indent
                ):
                    # Nested list
                    nested_ordered = bool(re.match(r"^\s*\d+[b]?\.\s", line))
                    i, nested_html = self._parse_list(lines, i, ordered=nested_ordered)
                    items[-1] += "\n" + nested_html
                    continue
                # Continuation of previous item
                items[-1] += " " + process_inline(line.strip())
                i += 1
                continue

            if indent < base_indent:
                break

            # List item at base level
            if ordered:
                m = re.match(r"^\s*\d+[b]?\.\s+(.*)", line)
            else:
                m = re.match(r"^\s*[-*+]\s+(.*)", line)

            if m:
                items.append(process_inline(m.group(1)))
                i += 1
            else:
                break

        parts = [f"<{tag}>"]
        for item in items:
            parts.append(f"<li>{item}</li>")
        parts.append(f"</{tag}>")
        return i, "\n".join(parts)

    @staticmethod
    def _is_list_item(line: str, ordered: bool, min_indent: int = 0) -> bool:
        """Check if a line is a list item."""
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent < min_indent:
            return False
        if ordered:
            return bool(re.match(r"\d+[b]?\.\s", stripped))
        return bool(re.match(r"[-*+]\s", stripped))

    def _parse_paragraph(self, lines: list[str], i: int) -> tuple[int, str]:
        """Parse a paragraph of text."""
        para_lines: list[str] = []
        while i < len(lines):
            line = lines[i]
            # Stop at blank lines, headings, code fences, HRs, lists, tables,
            # blockquotes, diagram markers
            if not line.strip():
                break
            if line.startswith("#"):
                break
            if re.match(r"^(`{3,}|~{3,})", line):
                break
            if re.match(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$", line.strip()):
                break
            if re.match(r"^\s*[-*+]\s", line):
                break
            if re.match(r"^\s*\d+[b]?\.\s", line):
                break
            if line.startswith(">"):
                break
            if re.match(r"^\s*<!--\s*diagram:", line):
                break
            if (
                "|" in line
                and i + 1 < len(lines)
                and re.match(r"^\s*\|?\s*[-:]+[-|:\s]+$", lines[i + 1])
            ):
                break
            para_lines.append(line)
            i += 1
        text = process_inline(" ".join(para_lines))
        return i, f"<p>{text}</p>"

    def _close_sections(self, html_text: str) -> str:
        """Insert </section> closing tags before each new <section> and at end."""
        parts = html_text.split('<section id="')
        if len(parts) <= 1:
            return html_text
        result = parts[0]
        for idx, part in enumerate(parts[1:]):
            if idx > 0:
                result += "</section>\n\n"
            result += '<section id="' + part
        result += "</section>"
        return result

    def build_toc(self) -> str:
        """Build the Table of Contents HTML from collected headings."""
        if not self.headings:
            return ""
        parts = [
            '<nav class="toc" aria-label="목차">',
            '<div class="toc-title">목차</div>',
            '<ol class="toc-list">',
        ]
        # Only include h2 and h3 in TOC
        in_sub = False
        for level, hid, text in self.headings:
            # Clean text for TOC display
            toc_text = process_inline(text)
            if level == 2:
                if in_sub:
                    parts.append("</ol>")
                    parts.append("</li>")
                    in_sub = False
                parts.append(f'<li><a href="#{hid}">{toc_text}</a>')
                # Check if next heading is h3 — we'll handle in iteration
                parts.append("</li>")
            elif level == 3:
                # Insert sub-list
                if not in_sub:
                    # Reopen the last <li>
                    if parts[-1] == "</li>":
                        parts.pop()
                    parts.append('<ol class="toc-list">')
                    in_sub = True
                parts.append(f'<li class="toc-h3"><a href="#{hid}">{toc_text}</a></li>')
        if in_sub:
            parts.append("</ol>")
            parts.append("</li>")
        parts.append("</ol>")
        parts.append("</nav>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------


def build_sidebar(active_page: str) -> str:
    """Build the sidebar navigation HTML."""
    parts = [
        '<nav class="sidebar" aria-label="문서 탐색">',
        '  <div class="sidebar-header">',
        '    <a href="index.html" class="sidebar-logo">',
        '      <svg viewBox="0 0 28 28" fill="none"'
        ' xmlns="http://www.w3.org/2000/svg" aria-hidden="true">',
        '        <rect width="28" height="28" rx="6" fill="#3182CE"/>',
        '        <path d="M7 14h4l2-4 3 8 2-4h3" stroke="#fff"'
        ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        "      </svg>",
        "      slm-factory",
        "    </a>",
        '    <span class="sidebar-version">Documentation</span>',
        "  </div>",
        '  <ul class="sidebar-nav">',
    ]
    for item in SIDEBAR_ITEMS:
        if item[0] == "section":
            parts.append(f'    <li class="nav-section">{item[1]}</li>')
        else:
            href = item[1]
            label = item[2]
            active_html = href.replace(".html", "")
            cls = ' class="active"' if active_html == active_page else ""
            parts.append(f'    <li><a href="{href}"{cls}>{label}</a></li>')
    parts.append("  </ul>")
    parts.append("</nav>")
    return "\n".join(parts)


def build_page(page_name: str, content_html: str, toc_html: str) -> str:
    """Assemble the full HTML page."""
    meta = PAGES[page_name]
    sidebar = build_sidebar(page_name)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meta["title"]} - slm-factory</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard-dynamic-subset.min.css">
  <link rel="stylesheet" href="./assets/style.css">
</head>
<body>
  <div class="page-wrapper">

    {sidebar}

    <div class="main-content">

      <header class="page-header">
        <div class="content-container">
          <span class="badge">{meta["badge"]}</span>
          <h1>{meta["title"]}</h1>
          <p class="subtitle">{meta["subtitle"]}</p>
        </div>
      </header>

      <div class="content-container">

        <nav class="breadcrumb" aria-label="경로">
          <a href="index.html">문서</a>
          <span class="separator" aria-hidden="true">/</span>
          <span>{meta["title"]}</span>
        </nav>

        {toc_html}

        {content_html}

      </div>
    </div>

    <footer class="page-footer">
      <p>slm-factory Documentation</p>
    </footer>

  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# SVG Extraction (one-time helper)
# ---------------------------------------------------------------------------


def extract_svgs() -> None:
    """Extract inline SVG diagrams from existing HTML files to docs/diagrams/."""
    diagrams_dir = DOCS_DIR / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    files_to_scan = [
        "guide.html",
        "architecture.html",
        "configuration.html",
        "quick-reference.html",
        "integration-guide.html",
    ]

    svg_re = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL)

    for fname in files_to_scan:
        fpath = DOCS_DIR / fname
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")
        page = fname.replace(".html", "")
        diagram_idx = 0

        for m in svg_re.finditer(content):
            svg_text = m.group()
            # Skip sidebar logos
            if 'viewBox="0 0 28 28"' in svg_text:
                continue
            diagram_idx += 1

            # Try to get a meaningful ID
            label_m = re.search(r'aria-labelledby="([^"]+)"', svg_text)
            if label_m:
                label_id = label_m.group(1).split()[0].replace("-title", "")
            else:
                label_id = f"diagram-{diagram_idx}"

            out_name = f"{page}-{label_id}.svg"
            out_path = diagrams_dir / out_name
            out_path.write_text(
                f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_text}',
                encoding="utf-8",
            )
            print(f"  Extracted: {out_name} ({len(svg_text):,} bytes)")

    print("SVG extraction complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if "--extract" in sys.argv:
        print("Extracting SVG diagrams from existing HTML files...")
        extract_svgs()
        return

    t0 = time.time()
    diagrams = load_diagrams()
    print(f"Loaded {len(diagrams)} diagram(s)")

    generated = 0
    skipped = 0
    for page_name, meta in PAGES.items():
        md_path = DOCS_DIR / f"{page_name}.md"
        html_path = DOCS_DIR / f"{page_name}.html"

        if not md_path.exists():
            print(f"  SKIP {page_name}.md (not found)")
            skipped += 1
            continue

        md_text = md_path.read_text(encoding="utf-8")
        converter = MarkdownConverter(diagrams=diagrams)
        content_html = converter.convert(md_text)
        toc_html = converter.build_toc()
        full_html = build_page(page_name, content_html, toc_html)

        html_path.write_text(full_html, encoding="utf-8")
        generated += 1
        print(
            f"  Built {page_name}.html"
            f" ({len(converter.headings)} headings,"
            f" {full_html.count('<table>'):,} tables,"
            f" {full_html.count('<pre'):,} code blocks)"
        )

    elapsed = time.time() - t0
    print(
        f"\nBuild summary: {generated} files generated, {skipped} skipped, {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
