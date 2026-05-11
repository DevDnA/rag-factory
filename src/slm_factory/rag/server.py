"""FastAPI RAG 서버 — Qdrant 검색 + Ollama 생성으로 RAG 응답을 제공합니다.

Qdrant에 적재된 벡터 DB를 검색하고, Ollama SLM에 컨텍스트와 함께
질문을 전달하여 문서 기반 답변을 생성하는 REST API 서버입니다.
"""

import os

# HF tokenizer fork 방지 + progress bar 노이즈 제거.
# SentenceTransformer/transformers import보다 먼저 설정되어야 합니다.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

import re

from ..utils import get_logger

logger = get_logger("rag.server")

# Ollama thinking 모델(Qwen3 등)의 <think> 태그를 제거합니다.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Gemma 등 모델의 특수 태그 (허용 목록 기반)
_SPECIAL_TAG_RE = re.compile(
    r"</?(?:end_of_turn|start_of_turn|channel|pad|eos|bos|sep|cls|mask|unk)\|?>",
    re.IGNORECASE,
)


def _clean_thinking_tags(text: str) -> str:
    text = _THINK_TAG_RE.sub("", text)
    text = _SPECIAL_TAG_RE.sub("", text)
    return text.strip()


def _sample_chunks_for_profile(
    *,
    qdrant_client: Any,
    collection_name: str,
    sample_size: int,
    strategy: str,
) -> tuple[list[str], list[str]]:
    """CorpusProfile 자동 생성을 위한 (표본 청크, 전체 청크 풀)을 Qdrant에서 추출.

    Returns
    -------
    tuple[list[str], list[str]]
        (sample_chunks, full_pool). ``sample_chunks``는 LLM 도메인 요약용 작은
        표본, ``full_pool``은 휴리스틱 약어 추출용 전체 청크. ``head`` 전략은
        sample과 동일한 청크만 보유하므로 둘이 같음.

    Parameters
    ----------
    strategy:
        - ``head``: 처음 sample_size개. full_pool은 같은 청크.
        - ``stratified``: 컬렉션을 균등 분할 — sample은 각 슬라이스 첫 청크,
          full_pool은 전체 docs.
        - ``random``: 무작위 sample, full_pool은 전체 docs.
    """
    import random

    info = qdrant_client.get_collection(collection_name=collection_name)
    total = info.points_count or 0
    if total <= 0:
        return [], []

    if strategy == "head":
        n = min(sample_size, total)
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=n,
            with_payload=True,
            with_vectors=False,
        )
        sample = [str(p.payload.get("document", "") or "") for p in points if p.payload]
        return sample, list(sample)

    # stratified / random — 모든 점을 scroll하여 표본 + 전체 풀 구성
    all_points: list = []
    offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    docs = [str(p.payload.get("document", "") or "") for p in all_points if p.payload]
    docs = [d for d in docs if d.strip()]
    if not docs:
        return [], []
    n = min(sample_size, len(docs))

    if strategy == "stratified":
        step = max(1, len(docs) // n)
        sample = [docs[i * step] for i in range(n)]
    else:  # random
        sample = random.sample(docs, n)

    return sample, docs


_RAG_SYSTEM_PROMPT = (
    "당신은 문서 기반 전문 어시스턴트입니다. "
    "아래 [참고 문서]를 근거로 질문에 답변하세요.\n\n"
    "규칙:\n"
    "1. 답변의 핵심 사실마다 어떤 문서에서 왔는지 명시하세요 (예: '문서 2에 따르면...').\n"
    "2. 여러 문서의 정보를 종합하되, 문서 간 내용이 상충하면 차이점을 명확히 설명하세요.\n"
    "3. 문서에 근거한 구체적 수치, 날짜, 명칭을 우선 사용하세요.\n"
    "4. 문서에 직접적 근거가 없으면 '해당 정보를 찾을 수 없습니다'라고 답변하세요.\n"
    "5. **역할·카테고리 단어도 유효한 답**: 문서가 특정 고유명사 대신 역할이나 카테고리(예: '지방자치단체', '발주기관', '공인시험기관')를 답으로 제시했다면 그것이 답입니다. 고유명사가 아니라는 이유로 '정보 부재' 처리 금지.\n"
    "6. **'X, Y 등'으로 함께 나열된 항목은 동일 카테고리의 예시**입니다. 임의로 다른 역할로 분리해 차이를 만들지 마세요. 차이가 명시되지 않았다면 '동일 역할의 예시이며 구체적 차이는 문서에 없습니다'라고 답변.\n"
    "7. 반드시 Markdown 문법으로 작성하세요. HTML 태그(<b>, <table>, <ul> 등)는 절대 사용하지 마세요. "
    "표는 | 구분자, 강조는 **굵게**, 목록은 - 또는 1. 형식을 사용하세요.\n"
    "8. 답변은 완결성 있게 작성하되 간결하게 핵심만 전달하세요."
)


def create_app(config: "SLMConfig"):
    """RAG API 서버를 위한 FastAPI 애플리케이션을 생성합니다.

    매개변수
    ----------
    config:
        slm-factory 프로젝트 설정. RAG, export, teacher 설정을 참조합니다.

    반환값
    -------
    FastAPI
        구성 완료된 FastAPI 애플리케이션 인스턴스.
    """
    import uuid
    from pathlib import Path

    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError:
        raise RuntimeError(
            "fastapi가 설치되지 않았습니다. uv sync --extra rag 로 설치하세요."
        )

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise RuntimeError(
            "qdrant-client가 설치되지 않았습니다. uv sync --extra rag 로 설치하세요."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers가 설치되지 않았습니다. "
            "uv sync --extra rag 로 설치하세요."
        )

    import httpx
    from pydantic import BaseModel, Field

    # ------------------------------------------------------------------
    # Pydantic 요청/응답 모델
    # ------------------------------------------------------------------

    class Source(BaseModel):
        """검색된 문서 소스 정보."""

        content: str
        doc_id: str
        score: float

    class QueryRequest(BaseModel):
        """RAG 질의 요청."""

        query: str = Field(..., min_length=1, max_length=4096)
        top_k: int | None = Field(None, ge=1, le=50)
        stream: bool = False

    class QueryResponse(BaseModel):
        """RAG 질의 응답."""

        answer: str
        sources: list[Source]
        query: str

    # ------------------------------------------------------------------
    # 설정값 추출
    # ------------------------------------------------------------------

    db_path = Path(config.paths.output) / config.rag.vector_db_path
    ollama_model = config.rag.ollama_model or config.teacher.model
    api_base = config.teacher.api_base
    rag_max_tokens = config.rag.max_tokens

    # ------------------------------------------------------------------
    # 라이프사이클 관리
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        qdrant_client = QdrantClient(path=str(db_path))
        collection_name = config.rag.collection_name
        try:
            count = qdrant_client.count(collection_name=collection_name).count
        except (ValueError, Exception):
            qdrant_client.close()
            raise RuntimeError(
                f"Qdrant 컬렉션 '{collection_name}'을(를) 찾을 수 없습니다.\n"
                f"  인덱스가 손상되었거나 구축이 완료되지 않았을 수 있습니다.\n"
                f"  해결: rm -rf {db_path} 후 slf rag를 다시 실행하세요."
            )
        if count == 0:
            qdrant_client.close()
            raise RuntimeError(
                f"Qdrant 컬렉션 '{collection_name}'이 비어 있습니다.\n"
                f"  해결: rm -rf {db_path} 후 slf rag를 다시 실행하세요."
            )
        logger.info(
            "Qdrant 컬렉션 로드 완료: %s (%d개 문서)",
            collection_name,
            count,
        )

        embedding_model = SentenceTransformer(config.rag.embedding_model)
        logger.info(
            "임베딩 모델 로드 완료: %s (device=%s)",
            config.rag.embedding_model,
            embedding_model.device.type,
        )

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.rag.request_timeout),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
        logger.info("Ollama 모델: %s, API: %s", ollama_model, api_base)

        _app.state.qdrant_client = qdrant_client
        _app.state.collection_name = collection_name
        _app.state.embedding_model = embedding_model
        _app.state.http_client = http_client
        _app.state.reranker = None
        _app.state.bm25_index = None
        _app.state.bm25_docs = None
        _app.state.bm25_ids = None
        _app.state.bm25_metadatas = None

        if config.rag.reranker_enabled:
            try:
                from sentence_transformers import CrossEncoder

                _app.state.reranker = CrossEncoder(
                    config.rag.reranker_model,
                    max_length=512,
                )
                logger.info("Reranker 모델 로드 완료: %s", config.rag.reranker_model)
            except Exception:
                logger.warning("Reranker 로드 실패 — 비활성 상태로 계속합니다")

        if config.rag.hybrid_search:
            try:
                all_points = []
                offset = None
                while True:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False,
                        offset=offset,
                    )
                    all_points.extend(points)
                    if next_offset is None:
                        break
                    offset = next_offset
                _app.state.bm25_docs = [
                    p.payload.get("document", "") for p in all_points
                ]
                _app.state.bm25_ids = [
                    p.payload.get("doc_id", str(p.id)) for p in all_points
                ]
                _app.state.bm25_metadatas = [
                    {
                        k: v
                        for k, v in p.payload.items()
                        if k not in ("document", "doc_id")
                    }
                    for p in all_points
                ]

                # BM25Okapi 인덱스 구축 (kiwi 형태소 토크나이저)
                try:
                    from rank_bm25 import BM25Okapi
                except ImportError:
                    raise RuntimeError(
                        "rank_bm25가 설치되지 않았습니다. "
                        "uv sync --extra rag 로 설치하세요."
                    )
                tokenized_corpus = [
                    _korean_tokenize(doc) for doc in _app.state.bm25_docs
                ]
                _app.state.bm25_index = BM25Okapi(tokenized_corpus)
                logger.info(
                    "BM25Okapi 인덱스 구축 완료: %d개 문서 (kiwi 형태소 토크나이저)",
                    len(_app.state.bm25_docs),
                )
            except Exception:
                logger.warning("BM25 인덱스 구축 실패 — 비활성 상태로 계속합니다")

        # -- CorpusProfile 로드/생성 ---
        # 인덱스가 어떤 도메인을 다루는지를 LLM이 자동 요약(또는 사용자 override)하여
        # IntentClassifier·합성 프롬프트에 컨텍스트로 주입합니다.
        _app.state.corpus_profile = None
        if config.rag.corpus_profile.enabled:
            from .corpus_profile import (
                CorpusProfile,
                generate_corpus_profile,
                load_corpus_profile,
                merge_with_override,
                save_corpus_profile,
            )

            cp_cfg = config.rag.corpus_profile
            profile_file = Path(config.paths.output) / cp_cfg.profile_path
            profile = load_corpus_profile(profile_file)

            # 자동 생성: 파일 없음 + auto_generate=True + Qdrant 컬렉션이 비어있지 않음.
            if profile.is_empty() and cp_cfg.auto_generate:
                try:
                    sample_chunks, full_pool = _sample_chunks_for_profile(
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        sample_size=cp_cfg.sample_size,
                        strategy=cp_cfg.sample_strategy,
                    )
                except Exception as exc:
                    logger.warning("CorpusProfile 청크 표본 추출 실패: %s", exc)
                    sample_chunks = []
                    full_pool = []

                if sample_chunks:
                    logger.info(
                        "CorpusProfile 자동 생성 시작 — sample %d청크 / 약어풀 %d청크 (%s)",
                        len(sample_chunks),
                        len(full_pool),
                        cp_cfg.sample_strategy,
                    )
                    profile = await generate_corpus_profile(
                        chunks=sample_chunks,
                        acronym_chunks=full_pool,
                        http_client=http_client,
                        ollama_model=ollama_model,
                        api_base=api_base,
                        request_timeout=min(config.rag.request_timeout, 120.0),
                        keep_alive=config.rag.agent.ollama_keep_alive,
                    )
                    if not profile.is_empty():
                        save_corpus_profile(profile, profile_file)
                        logger.info(
                            "CorpusProfile 자동 생성·저장: %s (keywords=%d)",
                            profile_file,
                            len(profile.keywords),
                        )

            # 사용자 override 병합 (모든 필드는 비어 있으면 자동 결과 유지).
            profile = merge_with_override(
                profile,
                name_override=cp_cfg.name,
                summary_override=cp_cfg.summary,
                keywords_override=cp_cfg.keywords if cp_cfg.keywords else None,
            )

            _app.state.corpus_profile = profile
            if not profile.is_empty():
                logger.info(
                    "CorpusProfile 로드: name=%r, keywords=%d개",
                    profile.name,
                    len(profile.keywords),
                )

        # -- Agent RAG 모드 초기화 ---
        _app.state.agent_session_manager = None
        _app.state.agent_tool_registry = None
        _cleanup_task = None

        if config.rag.agent.enabled:
            from .agent.session import SessionManager
            from .agent.state import FileBackedSessionStore
            from .agent.tools import ToolRegistry

            if config.rag.agent.persist_sessions:
                sessions_path = Path(config.paths.output) / config.rag.agent.sessions_dir
                _app.state.agent_session_manager = FileBackedSessionStore(
                    base_dir=sessions_path,
                    ttl=config.rag.agent.session_ttl,
                    max_turns=config.rag.agent.max_history_turns,
                )
                logger.info("Agent 세션 영속화 활성화: %s", sessions_path)
            else:
                _app.state.agent_session_manager = SessionManager(
                    ttl=config.rag.agent.session_ttl,
                    max_turns=config.rag.agent.max_history_turns,
                )
            _app.state.agent_tool_registry = ToolRegistry(
                app_state=_app.state,
                config=config,
                tokenize_fn=_korean_tokenize,
            )

            async def _cleanup_sessions():
                while True:
                    await asyncio.sleep(300)
                    try:
                        _app.state.agent_session_manager.cleanup_expired()
                    except Exception:
                        logger.warning("세션 정리 중 오류 발생", exc_info=True)

            _cleanup_task = asyncio.create_task(_cleanup_sessions())
            logger.info(
                "Agent RAG 모드 활성화 (max_iterations=%d, session_ttl=%ds)",
                config.rag.agent.max_iterations,
                config.rag.agent.session_ttl,
            )

        port = config.rag.server_port
        mode_label = "Agent RAG" if config.rag.agent.enabled else "RAG"
        logger.info(
            "\n\n"
            "  ╔══════════════════════════════════════════╗\n"
            "  ║  %s 채팅 서비스 준비 완료!              ║\n"
            "  ║  http://localhost:%d/chat              ║\n"
            "  ╚══════════════════════════════════════════╝\n",
            mode_label,
            port,
        )

        yield

        if _cleanup_task is not None:
            _cleanup_task.cancel()
        await http_client.aclose()
        qdrant_client.close()
        logger.info("RAG 서버 리소스 정리 완료")

    # ------------------------------------------------------------------
    # FastAPI 앱 생성
    # ------------------------------------------------------------------

    app = FastAPI(
        title="slm-factory RAG 서비스",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.rag.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_tracking(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s — %.3fs [%s]",
            request.method,
            request.url.path,
            elapsed,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("처리되지 않은 예외 [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "내부 오류가 발생했습니다. 서버 로그를 확인하세요.",
                "request_id": request_id,
            },
        )

    # ------------------------------------------------------------------
    # 엔드포인트
    # ------------------------------------------------------------------

    _kiwi_instance = None

    def _korean_tokenize(text: str) -> list[str]:
        nonlocal _kiwi_instance
        if _kiwi_instance is None:
            try:
                from kiwipiepy import Kiwi

                _kiwi_instance = Kiwi()
            except ImportError:
                _kiwi_instance = False
        if _kiwi_instance is False:
            return text.lower().split()
        tokens = _kiwi_instance.tokenize(text)  # type: ignore[union-attr]
        return [t.form for t in tokens if len(t.form) > 1 or not t.tag.startswith("J")]

    async def _rewrite_query(query: str) -> str:
        """짧은 질의를 검색에 적합한 형태로 확장합니다."""
        if len(query) > 30:
            return query
        http_client = app.state.http_client
        prompt = (
            f"다음 질의를 문서 검색에 적합하도록 확장하세요. "
            f"동의어, 관련 용어, 구체적 표현을 추가하세요. "
            f"확장된 질의만 반환하세요.\n\n"
            f"원본: {query}\n확장:"
        )
        try:
            response = await http_client.post(
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": config.rag.agent.ollama_keep_alive,
                    "options": {"num_predict": 100},
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                rewrite_data = response.json()
                raw = rewrite_data.get("response", "")
                if not raw:
                    raw = rewrite_data.get("thinking", "")
                expanded = _clean_thinking_tags(raw).strip()
                if expanded:
                    return f"{query} {expanded}"
        except Exception:
            logger.debug("Query rewriting 실패 — 원본 질의를 사용합니다")
        return query

    def _search_documents(body: QueryRequest):
        from .search import search_documents

        output = search_documents(
            body.query,
            top_k=body.top_k or config.rag.top_k,
            qdrant_client=app.state.qdrant_client,
            collection_name=app.state.collection_name,
            embedding_model=app.state.embedding_model,
            min_score=config.rag.min_score,
            reranker=app.state.reranker,
            hybrid_search=config.rag.hybrid_search,
            bm25_index=app.state.bm25_index,
            bm25_docs=getattr(app.state, "bm25_docs", None),
            bm25_ids=getattr(app.state, "bm25_ids", None),
            bm25_metadatas=getattr(app.state, "bm25_metadatas", None),
            tokenize_fn=_korean_tokenize,
        )

        sources = [
            Source(content=s.content, doc_id=s.doc_id, score=s.score)
            for s in output.sources
        ]
        context = "\n\n---\n\n".join(output.context_parts)
        prompt = (
            f"{_RAG_SYSTEM_PROMPT}\n\n"
            f"[참고 문서]\n{context}\n\n"
            f"질문: {body.query}\n답변:"
        )
        return sources, prompt

    @app.post("/query")
    async def query_rag(body: QueryRequest):
        """문서 검색 후 Ollama SLM으로 답변을 생성합니다.

        ``stream: true`` 요청 시 SSE(Server-Sent Events)로 토큰을 실시간 전송합니다.
        """
        from fastapi.responses import StreamingResponse

        if config.rag.query_rewriting:
            expanded = await _rewrite_query(body.query)
            body_for_search = QueryRequest(
                query=expanded, top_k=body.top_k, stream=body.stream
            )
        else:
            body_for_search = body

        sources, prompt = await asyncio.to_thread(_search_documents, body_for_search)
        http_client = app.state.http_client

        if body.stream:

            async def _generate_stream():
                has_response_tokens = False
                thinking_buf: list[str] = []
                try:
                    async with http_client.stream(
                        "POST",
                        f"{api_base}/api/generate",
                        json={
                            "model": ollama_model,
                            "prompt": prompt,
                            "stream": True,
                            "think": False,
                            "keep_alive": config.rag.agent.ollama_keep_alive,
                            "options": {
                                "num_predict": rag_max_tokens,
                            },
                        },
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            chunk = json.loads(line)
                            token = _SPECIAL_TAG_RE.sub("", chunk.get("response", ""))
                            if token:
                                has_response_tokens = True
                                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                            else:
                                thinking_token = chunk.get("thinking", "")
                                if thinking_token:
                                    thinking_buf.append(thinking_token)
                            if chunk.get("done"):
                                break
                except Exception as exc:
                    logger.error("Ollama 스트리밍 오류: %s", exc)
                    err_payload = json.dumps(
                        {"token": "\n\n[오류] 응답 생성 중 문제가 발생했습니다."},
                        ensure_ascii=False,
                    )
                    yield f"data: {err_payload}\n\n"

                # Ollama 0.19.0 호환: response가 빈 경우 thinking 내용을 fallback으로 전송
                if not has_response_tokens and thinking_buf:
                    fallback = _clean_thinking_tags("".join(thinking_buf))
                    if fallback:
                        yield f"data: {json.dumps({'token': fallback}, ensure_ascii=False)}\n\n"

                final = {
                    "sources": [s.model_dump() for s in sources],
                    "query": body.query,
                    "done": True,
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

            logger.info(
                "RAG 스트리밍 질의 시작 — 검색 %d건, 모델: %s",
                len(sources),
                ollama_model,
            )
            return StreamingResponse(
                _generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        import httpx

        try:
            response = await http_client.post(
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": config.rag.agent.ollama_keep_alive,
                    "options": {"num_predict": config.rag.max_tokens},
                },
            )
            response.raise_for_status()
        except httpx.ConnectError:
            logger.error("Ollama 서버에 연결할 수 없습니다: %s", api_base)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ollama_connection_error",
                    "message": (
                        f"Ollama 서버({api_base})에 연결할 수 없습니다. "
                        "Ollama가 실행 중인지 확인하세요."
                    ),
                },
            )
        except httpx.TimeoutException:
            logger.error("Ollama 응답 시간 초과 (timeout: %ss)", config.rag.request_timeout)
            return JSONResponse(
                status_code=504,
                content={
                    "error": "ollama_timeout",
                    "message": (
                        "Ollama 응답 시간이 초과되었습니다. "
                        "project.yaml의 rag.request_timeout 값을 늘려보세요."
                    ),
                },
            )
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP 오류 %d: %s", exc.response.status_code, exc.response.text[:200])
            return JSONResponse(
                status_code=502,
                content={
                    "error": "ollama_http_error",
                    "message": (
                        f"Ollama 서버 오류 (HTTP {exc.response.status_code}). "
                        f"모델 '{ollama_model}'이 올바르게 로드되었는지 확인하세요."
                    ),
                },
            )
        data = response.json()
        answer = _clean_thinking_tags(data.get("response", ""))
        if not answer:
            answer = _clean_thinking_tags(data.get("thinking", ""))

        logger.info(
            "RAG 질의 처리 완료 — 검색 %d건, 모델: %s",
            len(sources),
            ollama_model,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=body.query,
        )

    @app.get("/health/live")
    async def health_live() -> dict:
        """활성 상태 프로브 — 서버가 실행 중이면 항상 200."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready() -> dict:
        """준비 상태 프로브 — Qdrant 및 Ollama 연결을 확인합니다."""
        qdrant_client = app.state.qdrant_client
        collection_name = app.state.collection_name
        http_client = app.state.http_client

        status: dict = {
            "status": "ok",
            "qdrant": {"collection": config.rag.collection_name, "count": 0},
            "ollama": {"model": ollama_model, "status": "unknown"},
        }

        try:
            count_result = qdrant_client.count(collection_name=collection_name)
            status["qdrant"]["count"] = count_result.count
        except Exception:
            status["status"] = "degraded"
            status["qdrant"]["status"] = "error"

        try:
            resp = await http_client.get(
                f"{api_base}/api/tags",
                timeout=5.0,
            )
            resp.raise_for_status()
            status["ollama"]["status"] = "connected"
        except Exception:
            status["status"] = "degraded"
            status["ollama"]["status"] = "disconnected"

        return status

    @app.get("/health")
    async def health_check() -> dict:
        """하위 호환성을 위한 /health/ready 별칭."""
        return await health_ready()

    # ------------------------------------------------------------------
    # 웹 채팅 UI & SSE 스트리밍
    # ------------------------------------------------------------------

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page():
        html_path = Path(__file__).parent / "static" / "chat.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.post("/stream")
    async def query_stream(body: QueryRequest):
        if config.rag.query_rewriting:
            expanded = await _rewrite_query(body.query)
            body_for_search = QueryRequest(
                query=expanded, top_k=body.top_k, stream=body.stream
            )
        else:
            body_for_search = body

        sources, prompt = await asyncio.to_thread(_search_documents, body_for_search)
        http_client = app.state.http_client

        async def _generate():
            has_response_tokens = False
            thinking_buf: list[str] = []
            try:
                async with http_client.stream(
                    "POST",
                    f"{api_base}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": True,
                        "think": False,
                        "keep_alive": config.rag.agent.ollama_keep_alive,
                        "options": {
                            "num_predict": rag_max_tokens,
                        },
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = _SPECIAL_TAG_RE.sub("", chunk.get("response", ""))
                        if token:
                            has_response_tokens = True
                            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                        else:
                            thinking_token = chunk.get("thinking", "")
                            if thinking_token:
                                thinking_buf.append(thinking_token)
                        if chunk.get("done"):
                            break
            except Exception as exc:
                logger.error("Ollama 스트리밍 오류: %s", exc)
                err_payload = json.dumps(
                    {"type": "token", "content": "\n\n[오류] 응답 생성 중 문제가 발생했습니다."},
                    ensure_ascii=False,
                )
                yield f"data: {err_payload}\n\n"

            # Ollama 0.19.0 호환: response가 빈 경우 thinking 내용을 fallback으로 전송
            if not has_response_tokens and thinking_buf:
                fallback = _clean_thinking_tags("".join(thinking_buf))
                if fallback:
                    yield f"data: {json.dumps({'type': 'token', 'content': fallback}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        logger.info(
            "RAG SSE 스트리밍 시작 — 검색 %d건, 모델: %s",
            len(sources),
            ollama_model,
        )
        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Agent RAG 엔드포인트
    # ------------------------------------------------------------------

    class AgentQueryRequest(BaseModel):
        """Agent RAG 질의 요청."""

        query: str = Field(..., min_length=1, max_length=4096)
        session_id: str | None = Field(None, max_length=64)
        stream: bool = True

    class AgentQueryResponse(BaseModel):
        """Agent RAG 질의 응답."""

        answer: str
        sources: list[Source]
        query: str
        session_id: str
        iterations: int

    @app.get("/agent/status")
    async def agent_status():
        """Agent RAG 모드 활성화 상태를 반환합니다."""
        return {"enabled": config.rag.agent.enabled}

    @app.post("/agent")
    async def agent_query(body: AgentQueryRequest):
        """Agent RAG — stream 모드는 orchestrator를 통해, non-stream 모드는 AgentLoop.run()으로 처리합니다."""
        if not config.rag.agent.enabled:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "agent_mode_disabled",
                    "message": (
                        "Agent 모드가 비활성화되어 있습니다. "
                        "project.yaml에서 rag.agent.enabled: true로 설정하세요."
                    ),
                },
            )

        if body.stream:
            # Stream 모드 — orchestrator.handle_agent()가 세션/AgentLoop/planner 경로를 모두 처리.
            async def _sse_stream():
                async for event in _orchestrator.handle_agent(body.query, body.session_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-stream 모드 — AgentLoop.run()의 별도 코드 경로를 유지합니다
        # (run_stream과 force-answer 처리 방식이 달라 단순 교체가 위험).
        from .agent.loop import AgentLoop
        from .agent.session import Message

        session_mgr = app.state.agent_session_manager
        tool_registry = app.state.agent_tool_registry
        http_client = app.state.http_client

        session_id, _ = session_mgr.get_or_create(body.session_id)
        history = session_mgr.format_history(session_id)

        agent = AgentLoop(
            http_client=http_client,
            tool_registry=tool_registry,
            ollama_model=ollama_model,
            api_base=api_base,
            max_iterations=config.rag.agent.max_iterations,
            max_tokens=rag_max_tokens,
            request_timeout=config.rag.request_timeout,
            keep_alive=config.rag.agent.ollama_keep_alive,
        )
        session_mgr.add_message(session_id, Message(role="user", content=body.query))

        try:
            result = await agent.run(body.query, history)
        except Exception as exc:
            logger.error("Agent 실행 오류: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"error": "agent_error", "message": "Agent 실행 중 내부 오류가 발생했습니다."},
            )

        session_mgr.add_message(
            session_id, Message(role="assistant", content=result.answer)
        )

        sources = [
            Source(
                content=s.get("content", ""),
                doc_id=s.get("doc_id", ""),
                score=s.get("score", 0.0),
            )
            for s in result.sources
        ]

        return AgentQueryResponse(
            answer=result.answer,
            sources=sources,
            query=body.query,
            session_id=session_id,
            iterations=result.iterations,
        )

    # ------------------------------------------------------------------
    # 자동 라우팅 엔드포인트
    # ------------------------------------------------------------------

    from .agent.orchestrator import AgentOrchestrator
    from .agent.router import QueryRouter

    # IntentClassifier는 선택적 — 활성화 시 router에 주입.
    _intent_classifier = None
    if config.rag.agent.enabled and config.rag.agent.intent_classifier_enabled:
        from .agent.intent_classifier import IntentClassifier

        # Phase 9: router_model 슬롯이 있으면 사용, 없으면 기본 ollama_model.
        _router_model = (
            getattr(config.rag.agent.models, "router_model", "").strip()
            or ollama_model
        )

        # http_client는 lifespan에서 설정되므로 지연 접근. 여기서는 router가
        # route_async() 호출 시점에 app.state.http_client를 사용하도록 래퍼로 감쌈.
        # asyncio.Lock으로 동시 첫 호출에서 race-by-construction을 막습니다 —
        # 여러 요청이 동시에 도착해도 단일 IntentClassifier 인스턴스만 생성됩니다.
        # Lock은 이벤트 루프에 바인딩되므로 첫 async 호출 시점에 lazy 생성합니다.
        _agent_keep_alive = config.rag.agent.ollama_keep_alive

        class _LazyIntentClassifier:
            def __init__(self):
                self._cached = None
                self._lock: "asyncio.Lock | None" = None

            async def _get(self):
                if self._cached is not None:
                    return self._cached
                if self._lock is None:
                    self._lock = asyncio.Lock()
                async with self._lock:
                    if self._cached is None:
                        self._cached = IntentClassifier(
                            http_client=app.state.http_client,
                            ollama_model=_router_model,
                            api_base=api_base,
                            request_timeout=min(
                                config.rag.request_timeout, 10.0
                            ),
                            cache_ttl=(
                                config.rag.agent.intent_classifier_cache_ttl
                            ),
                            keep_alive=_agent_keep_alive,
                            corpus_profile=getattr(
                                app.state, "corpus_profile", None
                            ),
                        )
                return self._cached

            async def classify(self, query):
                inst = await self._get()
                return await inst.classify(query)

        _intent_classifier = _LazyIntentClassifier()

    _query_router = QueryRouter(
        agent_enabled=config.rag.agent.enabled,
        intent_classifier=_intent_classifier,
    )

    async def _simple_auto_stream(query: str):
        """단순 RAG 경로 — query_rewriting, 검색, Ollama 스트리밍을 이벤트 dict로 변환합니다."""
        if config.rag.query_rewriting:
            expanded = await _rewrite_query(query)
            body_for_search = QueryRequest(query=expanded, top_k=None, stream=True)
        else:
            body_for_search = QueryRequest(query=query, top_k=None, stream=True)

        sources, prompt = await asyncio.to_thread(_search_documents, body_for_search)
        http_client = app.state.http_client

        has_response_tokens = False
        thinking_buf: list[str] = []
        try:
            async with http_client.stream(
                "POST",
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
                    "keep_alive": config.rag.agent.ollama_keep_alive,
                    "options": {"num_predict": rag_max_tokens},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = _SPECIAL_TAG_RE.sub("", chunk.get("response", ""))
                    if token:
                        has_response_tokens = True
                        yield {"type": "token", "content": token}
                    else:
                        thinking_token = chunk.get("thinking", "")
                        if thinking_token:
                            thinking_buf.append(thinking_token)
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.error("Auto-simple 스트리밍 오류: %s", exc)
            yield {
                "type": "token",
                "content": "[오류] 응답 생성 중 문제가 발생했습니다.",
            }

        if not has_response_tokens and thinking_buf:
            fallback = _clean_thinking_tags("".join(thinking_buf))
            if fallback:
                yield {"type": "token", "content": fallback}

        yield {"type": "sources", "sources": [s.model_dump() for s in sources]}
        yield {"type": "done"}

    _orchestrator = AgentOrchestrator(
        router=_query_router,
        app_state=app.state,
        config=config,
        ollama_model=ollama_model,
        api_base=api_base,
        rag_max_tokens=rag_max_tokens,
        simple_stream_fn=_simple_auto_stream,
    )

    class AutoQueryRequest(BaseModel):
        """자동 라우팅 질의 요청."""

        query: str = Field(..., min_length=1, max_length=4096)
        session_id: str | None = Field(None, max_length=64)

    @app.post("/auto")
    async def auto_query(body: AutoQueryRequest):
        """질문 복잡도를 판단하여 Simple RAG 또는 Agent RAG로 자동 라우팅합니다."""

        async def _sse_stream():
            async for event in _orchestrator.handle_auto(body.query, body.session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            _sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # OpenAI 호환 엔드포인트 (OpenWebUI / 기타 OpenAI 클라이언트용)
    # ------------------------------------------------------------------

    class _OpenAIChatMessage(BaseModel):
        role: str
        content: str | list[dict] = ""

        def text(self) -> str:
            if isinstance(self.content, list):
                parts = []
                for item in self.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "".join(parts)
            return self.content or ""

    class _OpenAIChatRequest(BaseModel):
        model: str = "slm-factory-auto"
        messages: list[_OpenAIChatMessage]
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
        user: str | None = None

    _MODEL_AUTO = "slm-factory-auto"
    _MODEL_AGENT = "slm-factory-agent"
    _MODEL_RAG = "slm-factory-rag"
    _MODEL_ALIASES = {_MODEL_AUTO, _MODEL_AGENT, _MODEL_RAG}

    def _format_sources_markdown(sources: list[dict]) -> str:
        if not sources:
            return ""
        lines = ["\n\n---\n**📚 참고 문서**\n"]
        for i, s in enumerate(sources[:5], 1):
            doc_id = s.get("doc_id", "?")
            score = float(s.get("score", 0.0) or 0.0)
            preview = (s.get("content", "") or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "…"
            lines.append(f"{i}. `{doc_id}` · score {score:.2f} — {preview}")
        return "\n".join(lines)

    def _select_event_stream(model_id: str, query: str, session_id: str | None):
        resolved = model_id if model_id in _MODEL_ALIASES else _MODEL_AUTO
        if resolved == _MODEL_AGENT and config.rag.agent.enabled:
            return _orchestrator.handle_agent(query, session_id), resolved
        if resolved == _MODEL_RAG:
            return _simple_auto_stream(query), resolved
        return _orchestrator.handle_auto(query, session_id), resolved

    @app.get("/v1/models")
    async def openai_list_models():
        """OpenAI 호환 모델 목록 — OpenWebUI가 selector에 노출합니다."""
        now = int(time.time())
        data = [
            {
                "id": _MODEL_AUTO,
                "object": "model",
                "created": now,
                "owned_by": "slm-factory",
                "description": "Auto-routed (Simple RAG ↔ Agent RAG)",
            },
            {
                "id": _MODEL_RAG,
                "object": "model",
                "created": now,
                "owned_by": "slm-factory",
                "description": "Simple RAG (single-shot retrieval)",
            },
        ]
        if config.rag.agent.enabled:
            data.append(
                {
                    "id": _MODEL_AGENT,
                    "object": "model",
                    "created": now,
                    "owned_by": "slm-factory",
                    "description": "Agent RAG (multi-step ReAct)",
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(body: _OpenAIChatRequest, request: Request):
        """OpenAI Chat Completions 호환 엔드포인트.

        - 마지막 user 메시지를 query로 사용합니다.
        - 모델명으로 경로를 라우팅: ``slm-factory-auto`` | ``-agent`` | ``-rag``.
        - ``user`` 필드가 있으면 세션 키로 사용해 Agent RAG 히스토리를 유지합니다.
        """
        user_msgs = [m for m in body.messages if m.role == "user"]
        if not user_msgs:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "messages에 user role이 최소 1개 필요합니다.",
                        "type": "invalid_request_error",
                    }
                },
            )
        query = user_msgs[-1].text().strip()
        if not query:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "마지막 user 메시지가 비어 있습니다.",
                        "type": "invalid_request_error",
                    }
                },
            )

        session_key = body.user if body.user else None

        event_stream, resolved_model = _select_event_stream(
            body.model, query, session_key
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if body.stream:

            async def _openai_sse():
                first = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resolved_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

                sources_collected: list[dict] = []
                finish_reason = "stop"
                try:
                    async for event in event_stream:
                        etype = event.get("type")
                        if etype == "token":
                            content = event.get("content", "")
                            if not content:
                                continue
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": resolved_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        elif etype == "sources":
                            sources_collected = event.get("sources", []) or []
                        elif etype == "done":
                            break
                except Exception as exc:
                    logger.exception("OpenAI 호환 스트리밍 오류: %s", exc)
                    err_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": resolved_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "\n\n[오류] 응답 생성 중 문제가 발생했습니다."
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    finish_reason = "error"

                if sources_collected:
                    footer = _format_sources_markdown(sources_collected)
                    if footer:
                        footer_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": resolved_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": footer},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(footer_chunk, ensure_ascii=False)}\n\n"

                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resolved_model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _openai_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        content_parts: list[str] = []
        sources_collected: list[dict] = []
        try:
            async for event in event_stream:
                etype = event.get("type")
                if etype == "token":
                    content_parts.append(event.get("content", "") or "")
                elif etype == "sources":
                    sources_collected = event.get("sources", []) or []
                elif etype == "done":
                    break
        except Exception as exc:
            logger.exception("OpenAI 호환 non-stream 오류: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "응답 생성 중 오류가 발생했습니다.",
                        "type": "internal_error",
                    }
                },
            )

        full = "".join(content_parts).strip()
        if sources_collected:
            full = f"{full}{_format_sources_markdown(sources_collected)}"

        prompt_tokens = sum(len(m.text()) for m in body.messages) // 4
        completion_tokens = max(1, len(full) // 4)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": resolved_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def run_server(config: "SLMConfig") -> None:
    """RAG API 서버를 실행합니다.

    매개변수
    ----------
    config:
        slm-factory 프로젝트 설정. ``config.rag.server_host``와
        ``config.rag.server_port``로 서버 바인딩 주소를 결정합니다.
    """
    import uvicorn

    app = create_app(config)
    logger.info(
        "RAG 서버 시작: %s:%d (workers=%d, log_level=%s)",
        config.rag.server_host,
        config.rag.server_port,
        config.rag.workers,
        config.rag.log_level,
    )
    uvicorn.run(
        app,
        host=config.rag.server_host,
        port=config.rag.server_port,
        workers=config.rag.workers,
        log_level=config.rag.log_level,
    )
