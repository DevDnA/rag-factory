"""웹 검색 도구 — DuckDuckGo HTML 엔드포인트 기반 (API 키 불필요).

코퍼스 외 일반 지식·시기 의존 질의에 대해 LLM이 추측하지 않고 외부 사실을
조회할 수 있도록 합니다. 기본 백엔드는 DDG HTML(``https://html.duckduckgo.com/html``)
이며 무료·무인증으로 즉시 동작합니다.

설계 원칙
---------
- **never raise**: 네트워크·파싱 실패 시 빈 결과 반환 — 호출 측이 graceful
  degradation. 답변 품질 낮아질 뿐 서비스 중단 없음.
- **HTML 파싱은 보수적**: DDG HTML 구조 변경에 대비해 가능한 한 단순한 정규식
  추출만 사용. 추가 의존성(BS4) 없음.
- **stateless**: 캐시·쿠키 없음. 서버 lifespan의 ``http_client``를 그대로 사용.
"""

from __future__ import annotations

import html as html_module
import re
from dataclasses import dataclass
from typing import Any

from ...utils import get_logger

logger = get_logger("rag.agent.web_search")


_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# DDG HTML 응답에서 결과 단위 추출 — class="result__title"의 a 태그.
# 텍스트만 비파괴적으로 가져오기 위해 inner HTML 안의 모든 태그를 strip.
_RESULT_RE = re.compile(
    r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a\s+[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_DDG_REDIRECT_RE = re.compile(r"^//duckduckgo\.com/l/\?uddg=([^&]+)")


@dataclass(frozen=True)
class WebSearchResult:
    """단일 웹 검색 결과."""

    title: str
    url: str
    snippet: str


def _strip_html(text: str) -> str:
    """HTML 태그를 제거하고 엔터티 디코드, 공백 압축."""
    text = _TAG_RE.sub("", text)
    text = html_module.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _normalize_ddg_url(url: str) -> str:
    """DDG HTML이 반환하는 redirect URL(``//duckduckgo.com/l/?uddg=...``)에서 실제 URL 추출."""
    import urllib.parse

    if url.startswith("//"):
        url = "https:" + url
    match = _DDG_REDIRECT_RE.match(url[len("https:"):]) if url.startswith("https://") else None
    if match:
        try:
            return urllib.parse.unquote(match.group(1))
        except Exception:
            pass
    return url


def _parse_ddg_html(body: str, max_results: int) -> list[WebSearchResult]:
    """DDG HTML 응답에서 결과 목록을 추출합니다."""
    results: list[WebSearchResult] = []
    for match in _RESULT_RE.finditer(body):
        if len(results) >= max_results:
            break
        url = _normalize_ddg_url(match.group(1).strip())
        title = _strip_html(match.group(2))
        snippet = _strip_html(match.group(3))
        if not url or not title:
            continue
        results.append(WebSearchResult(title=title, url=url, snippet=snippet))
    return results


async def web_search(
    query: str,
    *,
    http_client: Any,
    max_results: int = 5,
    timeout: float = 8.0,
    region: str = "kr-kr",
) -> list[WebSearchResult]:
    """DuckDuckGo HTML 엔드포인트로 웹 검색 — never raises.

    Parameters
    ----------
    query:
        검색 쿼리 텍스트.
    http_client:
        재사용할 ``httpx.AsyncClient``.
    max_results:
        반환할 최대 결과 수.
    timeout:
        요청 타임아웃(초).
    region:
        DDG region 파라미터 (kr-kr, us-en 등). 한국어 질의는 kr-kr가 적합.
    """
    if not query or not query.strip():
        return []
    try:
        response = await http_client.post(
            _DDG_HTML_URL,
            data={"q": query, "kl": region},
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.text
    except Exception as exc:
        logger.warning("DDG 검색 실패: %s", exc)
        return []

    return _parse_ddg_html(body, max_results)


def format_results_for_prompt(
    results: list[WebSearchResult], *, snippet_max_chars: int = 240
) -> str:
    """LLM 프롬프트 컨텍스트로 주입할 텍스트 포맷."""
    if not results:
        return "(웹 검색 결과 없음)"
    lines: list[str] = ["[웹 검색 결과]"]
    for i, r in enumerate(results, start=1):
        snippet = r.snippet[:snippet_max_chars]
        if len(r.snippet) > snippet_max_chars:
            snippet += "..."
        lines.append(f"[웹 {i}] {r.title}")
        if r.url:
            lines.append(f"  URL: {r.url}")
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines)


__all__ = ["WebSearchResult", "format_results_for_prompt", "web_search"]
