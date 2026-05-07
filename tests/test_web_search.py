"""web_search 모듈 테스트 — DDG HTML 파서 + 결과 포맷."""

from __future__ import annotations

import pytest

from slm_factory.rag.agent.web_search import (
    WebSearchResult,
    _normalize_ddg_url,
    _parse_ddg_html,
    _strip_html,
    format_results_for_prompt,
    web_search,
)


class _FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttp:
    def __init__(self, response: _FakeResponse | None, *, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.last_url: str | None = None
        self.last_data: dict | None = None

    async def post(self, url, data=None, headers=None, timeout=None):
        self.last_url = url
        self.last_data = data
        if self._exc is not None:
            raise self._exc
        return self._response


_DDG_SAMPLE_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">First result</a>
  <a class="result__snippet" href="#">First snippet text</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Second &amp; result</a>
  <a class="result__snippet" href="#">Second snippet <b>bold</b></a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fc">Third</a>
  <a class="result__snippet" href="#">Third snippet</a>
</div>
</body></html>
"""


class TestParse:
    def test_strip_html_제거_엔터티_디코드(self):
        assert _strip_html("<b>Hello</b> &amp; world") == "Hello & world"

    def test_strip_html_공백_압축(self):
        assert _strip_html("a   b\n\nc") == "a b c"

    def test_normalize_ddg_redirect_url(self):
        url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath"
        assert _normalize_ddg_url(url) == "https://example.com/path"

    def test_normalize_상대_url_without_redirect(self):
        # DDG가 직접 URL을 주는 경우(드물지만)
        url = "//example.com/direct"
        assert _normalize_ddg_url(url) == "https://example.com/direct"

    def test_parse_샘플_HTML(self):
        results = _parse_ddg_html(_DDG_SAMPLE_HTML, max_results=10)
        assert len(results) == 3
        assert results[0].title == "First result"
        assert results[0].url == "https://example.com/a"
        assert "First snippet" in results[0].snippet
        assert results[1].title == "Second & result"
        assert "bold" in results[1].snippet  # <b> 안의 텍스트

    def test_parse_max_results_제한(self):
        results = _parse_ddg_html(_DDG_SAMPLE_HTML, max_results=2)
        assert len(results) == 2


class TestFormat:
    def test_빈_결과는_안내_텍스트(self):
        assert format_results_for_prompt([]) == "(웹 검색 결과 없음)"

    def test_여러_결과_포맷(self):
        results = [
            WebSearchResult(title="T1", url="https://a.com", snippet="snippet 1"),
            WebSearchResult(title="T2", url="https://b.com", snippet="snippet 2"),
        ]
        text = format_results_for_prompt(results)
        assert "[웹 검색 결과]" in text
        assert "[웹 1] T1" in text
        assert "https://a.com" in text
        assert "snippet 1" in text
        assert "[웹 2] T2" in text


class TestSearch:
    @pytest.mark.asyncio
    async def test_정상_검색(self):
        http = _FakeHttp(_FakeResponse(_DDG_SAMPLE_HTML))
        results = await web_search("hello", http_client=http, max_results=5)
        assert len(results) == 3
        assert http.last_url and "html.duckduckgo.com" in http.last_url
        assert http.last_data["q"] == "hello"

    @pytest.mark.asyncio
    async def test_빈_쿼리는_빈_결과(self):
        http = _FakeHttp(_FakeResponse(""))
        results = await web_search("", http_client=http)
        assert results == []
        assert http.last_url is None  # 호출 안 함

    @pytest.mark.asyncio
    async def test_HTTP_예외는_빈_결과(self):
        http = _FakeHttp(None, exc=RuntimeError("network down"))
        results = await web_search("query", http_client=http)
        assert results == []

    @pytest.mark.asyncio
    async def test_max_results_제한(self):
        http = _FakeHttp(_FakeResponse(_DDG_SAMPLE_HTML))
        results = await web_search("hello", http_client=http, max_results=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_HTML_파싱_실패는_빈_결과(self):
        http = _FakeHttp(_FakeResponse("<html>plain text without results</html>"))
        results = await web_search("hello", http_client=http)
        assert results == []
