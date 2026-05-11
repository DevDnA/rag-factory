# RAG 서비스 가이드

SLM + RAG로 도메인 AI 서비스를 구축하는 종합 가이드

---

## 1. 개요

### SLM 단독 사용의 한계

slm-factory로 만든 도메인 특화 SLM은 특정 분야에서 뛰어난 성능을 발휘하지만, 모든 상황을 완벽하게 대응하기는 어렵습니다.

- **할루시네이션(Hallucination)**: 학습 데이터에 포함되지 않은 질문에 대해 그럴듯하지만 부정확한 답변을 생성할 수 있습니다.
- **최신 정보 부재**: 파인튜닝 시점 이후에 추가되거나 변경된 정보를 반영할 수 없습니다.
- **근거 제시 불가**: 답변의 출처를 명시할 수 없습니다.
- **문서 간 관계 누락**: 각 문서를 개별 처리하므로, 서로 다른 문서에 걸친 개념의 연결을 파악하지 못합니다.

### RAG가 보완하는 방법

**RAG(Retrieval-Augmented Generation)**는 사용자 질문에 관련 문서를 검색하여 컨텍스트로 제공합니다. 할루시네이션을 억제하고, 최신 정보를 반영하며, 출처를 명시할 수 있습니다.

> **비유**: SLM은 **전문가의 뇌**, RAG는 **참고 도서관**입니다. 전문가가 도서관에서 근거를 찾아 답변하면 — 그것이 SLM + RAG 조합입니다.

---

## 2. 핵심 기술 소개

### 내장 RAG 스택

slm-factory는 다음 기술 스택으로 RAG 서비스를 제공합니다.

| 구성 요소 | 기술 | 역할 |
|-----------|------|------|
| **벡터 DB** | Qdrant (embedded mode) | 문서 청크 임베딩 저장 및 유사도 검색 (하이브리드 검색 지원) |
| **임베딩 모델** | Qwen/Qwen3-Embedding-0.6B | 다국어(한국어 포함) 비대칭 인코딩 문서·질의 벡터 변환 |
| **SLM 추론** | Ollama | 도메인 특화 SLM 서빙 (파인튜닝된 모델) |
| **API 서버** | FastAPI | REST API 엔드포인트 제공, SSE 스트리밍 지원 |

질의가 들어오면 Qwen3-Embedding-0.6B로 임베딩 → Qdrant에서 유사 문서 검색 → Ollama SLM이 검색 결과를 컨텍스트로 답변을 생성합니다.

---

## 3. 활용 패턴

RAG는 검색 기법이지 답변 생성 모델이 아닙니다. 반드시 LLM(Teacher든 Student든)이 필요합니다. slm-factory는 두 가지 패턴을 권장합니다.

### 패턴 1: RAG + 베이스 모델 — `slf rag`

| 항목 | 내용 |
|------|------|
| **LLM** | Teacher 모델(기본 `qwen3.5:9b`) — 파인튜닝 없이 그대로 사용 |
| **장점** | 즉시 시작(30초), 파인튜닝 불필요, 문서 수 제한 없음 |
| **단점** | Teacher(9B)가 직접 추론하므로 비용↑, 도메인 용어 이해가 제한적 |
| **적합한 경우** | 문서 20건 미만, PoC, 빠른 검증 |
| **명령어** | `slf rag` |

> **모델 자동 선택**: `slf rag`는 파인튜닝된 Student 모델이 Ollama에 등록되어 있으면 자동으로 해당 모델을 사용합니다. Student가 없을 때만 Teacher로 폴백합니다.

### 패턴 2: RAG + 파인튜닝 SLM — `slf tune` (권장)

| 항목 | 내용 |
|------|------|
| **LLM** | 파인튜닝된 Student 모델(1B) — 도메인 답변 스타일 학습 |
| **장점** | 도메인 어투·형식 일관성 + 최신 정보 + 출처 제시, 9배 빠르고 1/9 비용 |
| **단점** | 파인튜닝에 30분~1시간 소요, 문서 20건 이상 권장 |
| **적합한 경우** | 프로덕션 서비스, 대규모 동시 사용자 |
| **명령어** | `slf tune` |

### 패턴 선택 기준

| 조건 | 추천 |
|------|------|
| 문서 20건 미만, 즉시 시작하고 싶다 | **패턴 1** — `slf rag` |
| 문서 20건 이상, 프로덕션 서비스 | **패턴 2** — `slf tune` |

**권장**: **패턴 1로 즉시 시작**하고, 문서가 쌓이면 **패턴 2로 전환**하십시오. 두 패턴 모두 RAG 검색 결과를 근거로 답변하므로 할루시네이션이 억제됩니다.

### ⚠️ SLM 단독 사용에 대하여

파인튜닝된 모델을 RAG 없이 직접 사용(`ollama run`)할 수는 있지만, **프로덕션에서는 권장하지 않습니다.**

1B 모델의 파인튜닝은 **"답변 스타일"**을 학습시키는 것이지, **"도메인 지식"**을 내재화하는 것이 아닙니다. 파인튜닝으로 모델은 도메인에 맞는 어투와 형식을 배우지만, "제3조 예외 조항은 A, B, C이다" 같은 구체적 사실을 안정적으로 기억하지 못합니다. RAG 없이는 그럴듯하지만 틀린 답변을 자신 있게 생성합니다.

| 항목 | 수치 |
|------|------|
| 1B 모델 할루시네이션율 (RAG 없이) | >50% |
| RAG 적용 시 할루시네이션 감소 | 73~86% |
| 9B→1B 증류 시 능력 손실 | 40~60% |
| 업계 1B 단독 프로덕션 배포 비율 | <3% |

오프라인·격리 환경에서 불가피하게 SLM 단독 사용이 필요한 경우:

- 문서 50건 이상(QA 3,000개+)으로 충분히 학습시키십시오
- `slf eval run`으로 품질을 반드시 검증하십시오 (BLEU ≥ 0.3, ROUGE-L ≥ 0.4 권장)
- 답변에 "확인되지 않은 내용일 수 있습니다" 면책 문구를 포함하십시오

---

## 4. RAG 서비스 가이드

slm-factory는 문서 파싱 결과를 Qdrant에 인덱싱하고, FastAPI 기반 RAG API 서버로 즉시 서비스합니다.

```
export-corpus → corpus.parquet → rag-index → Qdrant → rag-serve → REST API
```

### 4.1 데이터 준비

```bash
# 1. slm-factory 파이프라인 실행 (파싱 + QA 생성)
slf tune

# 2. RAG 인덱싱용 코퍼스 내보내기
slf tool export-corpus

# 결과물:
#   output/corpus/corpus.parquet  — 코퍼스 데이터 (검색 대상 문서 청크)
#   output/corpus/qa.parquet      — QA 평가 데이터
```

코퍼스 데이터 (`corpus.parquet`):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `doc_id` | `str` | 문서 청크 고유 ID |
| `contents` | `str` | 청크 텍스트 내용 |
| `metadata` | `dict` | 메타데이터 |

평가용 QA 데이터 (`qa.parquet`):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `qid` | `str` | 질문 고유 ID |
| `query` | `str` | 질문 텍스트 |
| `retrieval_gt` | `list[list[str]]` | 정답 근거 문서 ID 목록 |
| `generation_gt` | `list[str]` | 정답 텍스트 목록 |

### 4.2 RAG 서비스 구동

slm-factory에 내장된 **Qdrant 인덱싱 + FastAPI 서빙**으로 즉시 RAG 서비스를 구동할 수 있습니다.

```bash
# 1. corpus.parquet 생성 (코퍼스 내보내기 완료 후)
slf tool export-corpus

# 2. Qdrant에 벡터 임베딩 적재
slf tool rag-index

# 3. RAG API 서버 실행
slf rag
# → POST http://localhost:8000/v1/query       질의 엔드포인트 (stream: true로 SSE 가능)
# → POST http://localhost:8000/v1/stream      웹 채팅 UI 전용 SSE 스트리밍
# → GET  http://localhost:8000/chat            내장 웹 채팅 UI
# → GET  http://localhost:8000/health          /health/ready 별칭 (Qdrant+Ollama 연결 확인)
# → GET  http://localhost:8000/health/ready    Qdrant+Ollama 연결 확인 (로드밸런서용)
# → GET  http://localhost:8000/health/live     라이브니스 체크 (항상 200)
```

> **팁**: `slf tune --chat` 명령으로 전체 파이프라인 실행 후 RAG 서버까지 한 번에 시작할 수 있습니다. 서버는 foreground로 실행되며, `Ctrl+C`로 종료합니다.

웹 채팅 UI(`http://localhost:8000/`)에는 다크/라이트 테마 토글, 추론 표시 토글, 모델 선택기(`auto` / `rag` / `agent`)가 있습니다.

```bash
# API 호출 테스트
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "도메인 특화 질문", "top_k": 5}'

# SSE 스트리밍 (토큰 단위 실시간 전송)
curl -N -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "도메인 특화 질문", "stream": true}'
```

**동작 방식:**
- `rag-index`: `corpus.parquet` 문서를 `Qwen/Qwen3-Embedding-0.6B`로 임베딩 → Qdrant에 적재
- `serve`: 질의 임베딩 → Qdrant 유사도 검색 → Ollama SLM 생성 → JSON 응답
- `stream: true` 요청 시 SSE(Server-Sent Events)로 토큰을 실시간 전송 (TTFT < 0.5초)

**질의 → 응답 흐름:**

<!-- diagram: integration-guide-diagram-1 -->

```
사용자 질의: "이 규정의 예외 조항은?"
       │
       ▼
  ① POST /v1/query ──────── FastAPI (rag-serve)
       │
       ▼
  ② 질의 임베딩 ─────────── Qwen3-Embedding-0.6B로 벡터 변환
       │
       ▼
  ③ Qdrant 검색 ───────────── cosine 유사도 top_k개 문서 청크 반환
       │
       ▼
  ④ 프롬프트 조합 ────────── 시스템 지시 + 검색 문서 + 질문 결합
       │                     "다음 문서를 참고하여 답변하십시오.
       │                      {검색된 문서 청크들}
       │                      질문: 이 규정의 예외 조항은?"
       ▼
  ⑤ Ollama SLM 생성 ──────── 도메인 특화 SLM이 컨텍스트 기반 답변 생성
       │
       ▼
  ⑥ JSON 응답
       {
         "answer": "제3조에 따르면 예외 조항은...",
         "sources": [
           {"content": "제3조 예외...", "doc_id": "chunk_03", "score": 0.92},
           {"content": "관련 규정...", "doc_id": "chunk_07", "score": 0.85}
         ],
         "query": "이 규정의 예외 조항은?"
       }
```

**적합한 경우:** PoC, 데모, 소규모 사내 서비스. 즉시 시작하여 API 서비스를 제공하고 싶을 때.

#### OpenAI 호환 엔드포인트 (OpenWebUI 연동)

RAG 서버는 OpenAI Chat Completions 호환 엔드포인트를 제공합니다. OpenWebUI, LibreChat 등 OpenAI API 클라이언트를 그대로 연결할 수 있습니다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/v1/models` | OpenWebUI selector에 노출되는 모델 목록 (`slm-factory-auto` / `-rag` / `-agent`) |
| `POST` | `/v1/chat/completions` | OpenAI 호환 채팅 — `stream: true`/`false`, multimodal `content` 리스트 수용 |

**모델 라우팅**

| 모델 ID | 동작 |
|---------|------|
| `slm-factory-auto` (기본값) | `AgentOrchestrator.handle_auto` — 질의 복잡도에 따라 simple/agent 자동 선택 |
| `slm-factory-rag` | 단일 패스 simple RAG |
| `slm-factory-agent` | Agent RAG (다단계 ReAct) — `rag.agent.enabled=true`일 때만 노출 |

알 수 없는 모델 ID는 `slm-factory-auto`로 fallback됩니다. `body.user` 필드가 있으면 세션 키로 사용되어 Agent RAG 히스토리를 유지합니다.

**API 호출 예시**

```bash
# 비스트리밍
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-factory-auto",
    "messages": [{"role": "user", "content": "도메인 질문"}],
    "stream": false
  }'

# 스트리밍 (OpenAI SSE 포맷)
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "slm-factory-agent",
    "messages": [{"role": "user", "content": "도메인 질문"}],
    "stream": true,
    "user": "session-abc"
  }'
```

스트리밍 응답은 표준 OpenAI 포맷(`data: {...}\n\n` + `data: [DONE]`)을 따르며, 답변 뒤에 참조 문서가 Markdown 표로 이어 붙습니다.

#### Agent RAG SSE 이벤트 타입 (`/auto` 및 `/agent` 경로)

`rag.agent.enabled: true`일 때, Agent Orchestrator는 ReAct 루프의 각 단계를 SSE 이벤트로 스트리밍합니다. 클라이언트는 다음 이벤트 타입을 처리해야 합니다:

| 이벤트 | 발행 시점 | payload 필드 |
|---|---|---|
| `route` | `/auto` 라우팅 직후 | `mode` (simple/agent), `intent?` (factual/comparative/...) |
| `thought` | ReAct 추론 단계 | `content` (추론 텍스트), `iteration` (반복 번호) |
| `action` | 도구 호출 | `content` (도구 이름), `input` (입력), `iteration` |
| `observation` | 도구 결과 | `content` (결과, 최대 `observation_preview_limit`=300자로 truncated), `iteration` |
| `token` | 답변 토큰 발행 | `content` (단일 토큰 또는 완전 답변) — planner 경로는 합성 완료 후 단일 이벤트로 발행 |
| `clarification` | ambiguous 질의 | `questions[]` (역질문 배열), `is_fallback` (bool) |
| `sources` | 참조 문서 | `sources[]` (문서 배열) — 빈 답변 시 생략 |
| `done` | 종료 | `session_id` (세션 ID) |

**예시**: 
```
data: {"type":"route","mode":"agent","intent":"factual"}
data: {"type":"thought","content":"사용자의 질문은 factual입니다. 검색을 시작합니다.","iteration":1}
data: {"type":"action","content":"search","input":"도메인 질문","iteration":1}
data: {"type":"observation","content":"검색 결과 1\n검색 결과 2\n...","iteration":1}
data: {"type":"token","content":"최종 답변 첫 번째 토큰"}
data: {"type":"token","content":" 이어서"}
data: {"type":"sources","sources":[{"doc_id":"chunk_01","content":"...","score":0.92}]}
data: {"type":"done","session_id":"user-abc-123"}
```

### 4.3 구축 단계 요약

```
Phase 1: slf tune (SLM 학습 + Ollama 배포)
    ↓
Phase 2: rag-index → Qdrant 인덱싱
    ↓
Phase 3: rag-serve → RAG API 서비스 운영
```

| 단계 | 기간 | 목표 | 판단 기준 |
|------|------|------|-----------|
| Phase 1 | 1-2주 | SLM 학습 완료 | BLEU ≥ 0.3, ROUGE-L ≥ 0.4 |
| Phase 2 | 1일 | 코퍼스 인덱싱 완료 | Qdrant 문서 적재 확인 |
| Phase 3 | 1-2주 | RAG 서비스 배포 | API 응답 품질 + 지연 시간 |

> **Phase 3 시점에서 실체적인 RAG 서비스가 동작합니다.** `slf tool rag-serve` 한 줄로 REST API를 제공하는 도메인 AI 서비스가 완성됩니다.

### 4.4 프로덕션 보완 체크리스트

slm-factory 내장 RAG 서버로 RAG 서비스를 즉시 운영할 수 있지만, **엔터프라이즈 프로덕션 환경**에서는 다음 항목을 보완해야 합니다.

#### 즉시 보완 (서비스 공개 전)

| 항목 | 설명 | 보완 방법 |
|------|------|-----------|
| **HTTPS/TLS** | API 통신 암호화 | Nginx/Caddy 리버스 프록시로 TLS 종단. `certbot`으로 인증서 자동 발급 |
| **인증/인가** | 허가된 사용자만 접근 | API 키 기반: Nginx `auth_request` 또는 FastAPI 미들웨어. 기업 환경: SSO/LDAP 연동 |
| **입력 검증** | 프롬프트 인젝션 방어 | 질의 길이 제한 (예: 2,000자), 금칙어 필터링, 시스템 프롬프트 고정 |
| **헬스체크** | 서비스 상태 모니터링 | 내장 `/health`(기본), `/health/ready`(Ollama+Qdrant 연결), `/health/live`(라이브니스 체크, `{"status": "ok"}` 응답) 엔드포인트 제공. 프로덕션에서는 `/health/ready`를 로드밸런서 헬스체크에, `/health/live`를 Kubernetes 라이브니스 프로브에 사용 |

#### 운영 안정화 (서비스 공개 후)

| 항목 | 설명 | 보완 방법 |
|------|------|-----------|
| **요청 제한** | 과부하 방지 | Nginx `limit_req_zone` 또는 FastAPI `slowapi`. 사용자별 분당 요청 수 제한 |
| **로깅** | 질의/응답 추적 | 구조화 로깅 (JSON). 질의, 검색된 문서 ID, 응답 시간, 에러를 기록 |
| **모니터링** | 성능·품질 추적 | Prometheus + Grafana: 응답 지연(p50/p95/p99), 에러율, Ollama 추론 시간 |
| **자동 재시작** | 장애 복구 | systemd `Restart=always` 또는 Docker `restart: unless-stopped` |
| **백업** | 데이터 보호 | 벡터 DB(Qdrant) 정기 백업. Qdrant 벡터 인덱스 버전 관리 |

#### 프로덕션 확장 (사용자 증가 시)

| 항목 | 설명 | 보완 방법 |
|------|------|-----------|
| **로드 밸런서** | 수평 확장 | 서버 인스턴스 다중 구동 + Nginx upstream으로 수평 확장 |
| **GPU 추론 최적화** | 높은 동시성 | Ollama 대신 vLLM 또는 TGI(Text Generation Inference)로 전환. 배치 추론 지원 |
| **응답 캐싱** | 반복 질의 최적화 | Redis 캐시. 동일 질의 재처리 방지 (TTL 기반) |
| **벡터 DB 확장** | 대규모 코퍼스 | Qdrant embedded → Qdrant 서버 모드 또는 Milvus/Weaviate. 분산 인덱싱, 수백만 문서 지원 |
| **모델 버전 관리** | SLM 업데이트 | `tool evolve`로 SLM 갱신 시, Ollama 모델명에 버전 태그 부여 (`v1`, `v2`). Blue-Green 배포로 무중단 전환 |

#### 프로덕션 확장 판단 기준

내장 RAG 서버에서 커스텀 확장이 필요한 시점:

| 신호 | 의미 |
|------|------|
| 동시 사용자 10명 이상에서 응답 지연 증가 | 서버 인스턴스 수평 확장 필요 |
| 인증·권한 로직이 복잡해짐 | 미들웨어 커스터마이징 필요 |
| 검색 결과 후처리 로직 필요 | 커스텀 비즈니스 로직 삽입 필요 |
| 멀티 모델 라우팅 필요 | 질문 유형별 다른 SLM 호출 |

내장 RAG 서버로 시작하여, 위 신호가 나타나면 Nginx 로드밸런서 추가, GPU 추론 최적화(vLLM/TGI), 벡터 DB 확장(Milvus/Weaviate) 등을 단계적으로 적용하십시오.

### 4.5 운영 팁

- **문서 분리 전략**: 핵심 도메인 지식은 SLM 학습에, 세부 참조 자료는 RAG 검색 코퍼스에 활용합니다.
- **모델 크기**: RAG와 함께 사용하면 1B~3B 소형 모델로도 충분합니다. SLM이 모든 지식을 기억할 필요 없이 "검색된 문서 이해"에 집중하면 됩니다.
- **시스템 프롬프트 통일**: slm-factory 학습 시 사용한 시스템 프롬프트와 RAG 서빙 시 프롬프트를 일관되게 유지하십시오.
- **업데이트**: slm-factory의 `tool evolve`로 SLM을 진화시키고, RAG 코퍼스에 새 문서를 추가하여 양쪽을 동시에 갱신합니다.

---

## 5. 활용 시나리오

- **사내 규정 Q&A**: 규정 용어를 이해하는 SLM + 최신 규정 문서를 검색하는 RAG. 정확한 조항과 출처를 제시합니다.
- **장애 대응 AI**: 장애 패턴을 인식하는 SLM + 유사 장애 사례와 복구 절차서를 검색하는 RAG.
- **의료/법률 자문**: 전문 용어를 정확히 사용하는 SLM + 관련 조문/판례 원문을 검색하는 RAG.

---

## 6. 주의사항

- **단순한 문서 세트**: FAQ나 단일 매뉴얼에서는 RAG 추가 효과가 미미합니다. 패턴 1(SLM 단독)로 충분합니다.
- **검색 품질 의존**: RAG 응답 품질은 검색 정확도에 직접 의존합니다. 코퍼스 청킹 전략과 임베딩 모델 선택이 중요합니다.
- **추가 비용**: RAG 검색과 벡터 DB 운영에 처리 시간과 인프라 비용이 추가됩니다.
- **운영 복잡도**: SLM + RAG 조합은 두 시스템을 함께 운영해야 합니다. 팀 역량과 인프라를 고려하십시오.

---

## 7. 관련 문서

| 문서 | 내용 |
|------|------|
| [사용 가이드](guide.md) | slm-factory 설치부터 모델 배포까지 단계별 안내 |
| [설정 레퍼런스](configuration.md) | `rag`, `export.ollama` 설정 상세 |
| [CLI 레퍼런스](cli-reference.md) | `tool rag-index`, `tool rag-serve` 명령어 상세 |
| [아키텍처 가이드](architecture.md) | 모듈 내부 구조와 설계 원칙 |
| [개발 가이드](development.md) | 모듈 확장 방법 |


