"""RAG 서버 통합 테스트 — FastAPI TestClient로 end-to-end 검증.

외부 의존성(Qdrant, Ollama, embedding 모델)은 모두 mock하여 실제 네트워크·
파일 I/O 없이 엔드포인트·라이프사이클·orchestrator 연결을 검증합니다.

설계 원칙
---------
- **단위 테스트의 공백을 메움**: 개별 모듈은 별도 테스트로 커버되며, 여기서는
  FastAPI 라이프사이클·SSE framing·app.state 와이어링 같은 "통합" 동작만 확인.
- **결정적 모킹**: 외부 호출은 고정된 응답을 반환 — 외부 인프라 없이 CI에서 실행.
- **최소 범위**: 각 엔드포인트의 golden path + 라우팅 결정 정도만 검증.
  세부 비즈니스 로직은 단위 테스트에서 다룹니다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 외부 의존성 Fake/Mock
# ---------------------------------------------------------------------------


class _FakeCountResult:
    count = 3


class _FakeQdrant:
    """실제 Qdrant 호출 없이 lifespan 통과시키는 stub."""

    def __init__(self, *args, **kwargs):
        pass

    def count(self, **kwargs):
        return _FakeCountResult()

    def scroll(self, **kwargs):
        return ([], None)

    def close(self):
        pass


class _FakeST:
    """SentenceTransformer stub — encode는 실제로 호출되지 않음 (search_documents를 mock하므로)."""

    def __init__(self, *args, **kwargs):
        self.device = SimpleNamespace(type="cpu")

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        # 실 호출 시에도 안전하게 동작하도록 0-벡터 반환
        import numpy as np

        if isinstance(texts, str):
            return np.zeros(8, dtype=np.float32)
        return np.zeros((len(texts), 8), dtype=np.float32)


class _FakeStreamContext:
    """httpx.AsyncClient.stream(...) 반환값을 흉내내는 async context manager."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    """httpx.AsyncClient 교체본 — Ollama 호출을 고정 응답으로 처리."""

    # 클래스 전역 스크립트 — 테스트가 이걸 조작.
    stream_tokens: list[str] = ["모킹된 ", "답변"]
    post_response: dict = {"response": "모킹된 비스트리밍 답변", "done": True}
    health_response: dict = {"models": []}

    def __init__(self, **kwargs):
        pass

    async def aclose(self):
        pass

    async def post(self, url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = dict(self.post_response)
        resp.raise_for_status = MagicMock()
        return resp

    async def get(self, url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = dict(self.health_response)
        resp.raise_for_status = MagicMock()
        return resp

    def stream(self, method: str, url: str, **kwargs):
        lines = [
            json.dumps({"response": tok, "done": False}, ensure_ascii=False)
            for tok in self.stream_tokens
        ] + [json.dumps({"response": "", "done": True})]
        return _FakeStreamContext(lines)


def _fake_search_documents(query, **kwargs):
    """search.search_documents 교체본 — 고정 sources 반환."""
    from slm_factory.rag.search import SearchOutput, SearchResult

    return SearchOutput(
        sources=[
            SearchResult(
                content=f"검색 결과: {query}",
                doc_id="doc_1",
                score=0.91,
                metadata={},
            )
        ],
        context_parts=[f"검색 결과: {query}"],
    )


# ---------------------------------------------------------------------------
# Fixture: 통합 테스트용 FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_client(monkeypatch, tmp_path):
    """모든 외부 의존성이 모킹된 FastAPI TestClient."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    # 외부 모듈 교체
    import httpx
    import qdrant_client
    import sentence_transformers

    monkeypatch.setattr(qdrant_client, "QdrantClient", _FakeQdrant)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    # search_documents를 결정적으로 교체 — 실제 임베딩/벡터 검색 우회.
    monkeypatch.setattr(
        "slm_factory.rag.search.search_documents", _fake_search_documents
    )

    # 테스트용 config — 무거운 기능은 비활성화.
    from slm_factory.config import SLMConfig

    config = SLMConfig.model_validate(
        {
            "project_name": "integration-test",
            "paths": {"output": str(tmp_path / "output")},
            "domain_docs": {"path": str(tmp_path / "docs")},
            "rag": {
                "reranker_enabled": False,
                "hybrid_search": False,
                "query_rewriting": False,
                "ollama_model": "test-model",
                "request_timeout": 30.0,
                "agent": {
                    "enabled": True,
                    "stream_reasoning": False,  # 테스트 이벤트 단순화
                    "planner_enabled": False,
                    "session_source_reuse": False,
                },
            },
        }
    )

    # 테스트 전에 스크립트 기본값 재설정
    _FakeAsyncClient.stream_tokens = ["모킹된 ", "답변"]
    _FakeAsyncClient.post_response = {"response": "모킹된 비스트리밍 답변", "done": True}

    from slm_factory.rag.server import create_app

    app = create_app(config)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# 헬스 체크 엔드포인트
# ---------------------------------------------------------------------------


class TestHealth:
    """라이프사이클이 정상적으로 완료되면 헬스체크가 작동합니다."""

    def test_live_엔드포인트(self, rag_client):
        response = rag_client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_엔드포인트(self, rag_client):
        response = rag_client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ok", "degraded")
        assert "qdrant" in body
        assert "ollama" in body

    def test_health_alias(self, rag_client):
        response = rag_client.get("/health")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /query 비스트리밍
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """/query 기본 동작."""

    def test_비스트리밍_응답(self, rag_client):
        response = rag_client.post("/query", json={"query": "테스트"})
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert body["query"] == "테스트"
        assert body["answer"] == "모킹된 비스트리밍 답변"
        assert len(body["sources"]) == 1
        assert body["sources"][0]["doc_id"] == "doc_1"

    def test_스트리밍_응답_SSE_형식(self, rag_client):
        response = rag_client.post(
            "/query", json={"query": "테스트", "stream": True}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        lines = [
            line
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert lines, "SSE 응답이 비어있습니다"

        # 마지막 이벤트는 done=True를 포함
        last = json.loads(lines[-1][len("data: "):])
        assert last.get("done") is True
        assert "sources" in last


# ---------------------------------------------------------------------------
# /stream 엔드포인트 — /chat UI용 SSE
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    """/stream은 /query와 유사하지만 이벤트 형식이 다릅니다."""

    def test_token_sources_done_이벤트_순서(self, rag_client):
        response = rag_client.post("/stream", json={"query": "테스트"})
        assert response.status_code == 200

        events = _parse_sse(response.text)
        types = [e.get("type") for e in events]

        # token 이벤트가 있어야 함
        assert "token" in types
        # sources 이벤트
        assert "sources" in types
        # 마지막은 done
        assert types[-1] == "done"


# ---------------------------------------------------------------------------
# /auto 자동 라우팅
# ---------------------------------------------------------------------------


class TestAutoEndpoint:
    """/auto는 라우터로 단순/agent 경로 결정 후 SSE 발행."""

    def test_단순_질의는_simple_route(self, rag_client):
        response = rag_client.post("/auto", json={"query": "오늘 날씨"})
        assert response.status_code == 200

        events = _parse_sse(response.text)
        assert events[0]["type"] == "route"
        assert events[0]["mode"] == "simple"
        assert events[-1]["type"] == "done"

    def test_복합_질의는_agent_route(self, rag_client):
        # _FakeAsyncClient는 모든 post/stream에 동일하게 응답하므로,
        # agent 경로라도 정상 종료만 확인
        response = rag_client.post(
            "/auto", json={"query": "A와 B의 차이 비교"}
        )
        assert response.status_code == 200

        events = _parse_sse(response.text)
        assert events[0]["type"] == "route"
        assert events[0]["mode"] == "agent"
        assert events[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# /agent 엔드포인트
# ---------------------------------------------------------------------------


class TestAgentEndpoint:
    """/agent stream 모드는 orchestrator.handle_agent를 통해 실행됩니다."""

    def test_agent_status(self, rag_client):
        response = rag_client.get("/agent/status")
        assert response.status_code == 200
        assert response.json() == {"enabled": True}

    def test_stream_모드_SSE_응답(self, rag_client):
        response = rag_client.post(
            "/agent", json={"query": "테스트", "stream": True}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(response.text)
        # /agent는 route 이벤트 없이 바로 agent 경로
        assert not any(e.get("type") == "route" for e in events)
        assert events[-1]["type"] == "done"
        assert "session_id" in events[-1]


# ---------------------------------------------------------------------------
# Phase 15a — smart_mode 프리셋 통합
# ---------------------------------------------------------------------------


@pytest.fixture
def smart_rag_client(monkeypatch, tmp_path):
    """smart_mode=True 프리셋으로 lifespan까지 통과하는 TestClient."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    import httpx
    import qdrant_client
    import sentence_transformers

    monkeypatch.setattr(qdrant_client, "QdrantClient", _FakeQdrant)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        "slm_factory.rag.search.search_documents", _fake_search_documents
    )

    from slm_factory.config import SLMConfig

    config = SLMConfig.model_validate(
        {
            "project_name": "smart-mode-test",
            "paths": {"output": str(tmp_path / "output")},
            "domain_docs": {"path": str(tmp_path / "docs")},
            "rag": {
                "reranker_enabled": False,
                "hybrid_search": False,
                "query_rewriting": False,
                "ollama_model": "test-model",
                "request_timeout": 30.0,
                "agent": {
                    "enabled": True,
                    "stream_reasoning": False,
                    "smart_mode": True,  # 원클릭 P0 전체 활성화
                },
            },
        }
    )

    _FakeAsyncClient.stream_tokens = ["답", "변"]
    _FakeAsyncClient.post_response = {
        # Planner/Verifier/Reflector/Reviewer가 모두 "OK" 또는 "factual" 반환하도록
        # JSON 다목적 응답
        "response": (
            '{"intent": "factual", "confidence": 0.9, "reason": "x",'
            ' "sufficient": true, "answer_ok": true,'
            ' "passed": true,'
            ' "strategy": "fact", "rationale": "ok",'
            ' "steps": [{"tool": "search", "args": {"query": "test"}, "reason": "r"}]}'
        ),
        "done": True,
    }

    from slm_factory.rag.server import create_app

    app = create_app(config)
    with TestClient(app) as client:
        yield client


class TestSmartModeIntegration:
    """smart_mode=True 프리셋이 실제 FastAPI 라이프사이클을 통과."""

    def test_smart_mode_lifespan_통과(self, smart_rag_client):
        response = smart_rag_client.get("/health/live")
        assert response.status_code == 200

    def test_smart_mode_auto_엔드포인트(self, smart_rag_client):
        response = smart_rag_client.post("/auto", json={"query": "테스트 질의"})
        assert response.status_code == 200
        events = _parse_sse(response.text)
        # route 이벤트 + 마지막 done
        assert events[0]["type"] == "route"
        # intent 필드가 추가됨 (IntentClassifier 활성)
        assert "intent" in events[0]
        assert events[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# OpenAI 호환 엔드포인트 (OpenWebUI)
# ---------------------------------------------------------------------------


class TestOpenAICompat:
    """/v1/models, /v1/chat/completions — OpenWebUI 연동용."""

    def test_models_목록(self, rag_client):
        response = rag_client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        ids = {m["id"] for m in body["data"]}
        assert "slm-factory-auto" in ids
        assert "slm-factory-rag" in ids
        # agent.enabled=True 이므로 agent 모델도 노출
        assert "slm-factory-agent" in ids

    def test_chat_completions_비스트리밍(self, rag_client):
        response = rag_client.post(
            "/v1/chat/completions",
            json={
                "model": "slm-factory-rag",
                "messages": [{"role": "user", "content": "테스트"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "slm-factory-rag"
        assert body["choices"][0]["message"]["role"] == "assistant"
        # 본문 + sources footer
        content = body["choices"][0]["message"]["content"]
        assert content  # 비어있지 않음
        assert "참고 문서" in content
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_chat_completions_스트리밍_SSE(self, rag_client):
        response = rag_client.post(
            "/v1/chat/completions",
            json={
                "model": "slm-factory-auto",
                "messages": [{"role": "user", "content": "오늘 날씨"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        chunks = [
            line[len("data: "):]
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert chunks, "SSE 응답이 비어있습니다"
        assert chunks[-1] == "[DONE]"

        parsed = [json.loads(c) for c in chunks if c != "[DONE]"]
        # 첫 청크는 role=assistant
        assert parsed[0]["choices"][0]["delta"].get("role") == "assistant"
        # 마지막 청크는 finish_reason=stop
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
        # object 포맷 확인
        assert all(p["object"] == "chat.completion.chunk" for p in parsed)
        # content delta가 하나 이상 있어야 함
        content_chunks = [
            p for p in parsed if p["choices"][0]["delta"].get("content")
        ]
        assert content_chunks

    def test_chat_completions_user_없는_요청_400(self, rag_client):
        response = rag_client.post(
            "/v1/chat/completions",
            json={
                "model": "slm-factory-auto",
                "messages": [{"role": "system", "content": "sys"}],
                "stream": False,
            },
        )
        assert response.status_code == 400
        assert "user" in response.json()["error"]["message"]

    def test_chat_completions_빈_query_400(self, rag_client):
        response = rag_client.post(
            "/v1/chat/completions",
            json={
                "model": "slm-factory-auto",
                "messages": [{"role": "user", "content": "   "}],
                "stream": False,
            },
        )
        assert response.status_code == 400

    def test_chat_completions_multimodal_content_list(self, rag_client):
        """OpenWebUI가 content를 list[dict]로 보낼 때도 텍스트만 추출."""
        response = rag_client.post(
            "/v1/chat/completions",
            json={
                "model": "slm-factory-rag",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "테스트"}],
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"]

    def test_chat_completions_unknown_model_fallback_auto(self, rag_client):
        """알 수 없는 모델명은 auto로 fallback."""
        response = rag_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4-whatever",
                "messages": [{"role": "user", "content": "오늘 날씨"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["model"] == "slm-factory-auto"


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict]:
    """SSE body를 JSON 이벤트 dict 목록으로 파싱합니다."""
    events: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[len("data: "):]))
        except json.JSONDecodeError:
            continue
    return events
