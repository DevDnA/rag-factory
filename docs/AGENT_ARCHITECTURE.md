# Agent RAG 아키텍처

oh-my-openagent (OMO) 디자인 패턴을 RAG Q&A 컨텍스트에 이식한 **똑똑한 agent 파이프라인**의 최종 구조 문서.

## 파이프라인 전경

```
사용자 질의
  │
  ▼
[Hook] pre_query        (query 정규화)
  │
  ▼
QueryRouter.route_async
  ├─ IntentClassifier (Phase 5)
  └─ 키워드 fallback
  │
  ▼ { intent, mode }
  │
  ├─ ambiguous + clarifier_enabled ──► Clarifier (Phase 11) ─► clarification 이벤트 ─► 종료
  ├─ simple mode ─────────────────────► 단순 RAG → 종료
  └─ agent mode
       │
       ▼
    PersonaRouter.select(intent)
      ├─ CustomPersona (Phase 14)     ← 우선
      └─ Built-in Persona (Phase 6)    ← Researcher/Comparator/Analyst/Procedural
       │
       ▼
    Planner (Phase 1c) ─► ExecutionPlan(strategy, steps)
       │
       ├─ plan.is_fallback + legacy_fallback_enabled ──► AgentLoop (ReAct, Phase 2)
       │
       ▼
    Plan step 실행
      ├─ parallel_steps + parallel_safe×N ──► asyncio.gather (Phase 3-b)
      └─ 직렬 실행
         │
         ▼
       ToolRegistry.execute → ToolResult.sources → all_sources
       │
       ▼
    Verifier (Phase 1c) ─► 부족하면 repair search (max N)
       │
       ▼
    [Hook] post_search     (dedup, boosting)
       │
       ▼
    _stream_synthesis
      ├─ Persona's synthesis_prompt_template 우선
      ├─ Skills addon 주입 (Phase 7)
      ├─ 이전 턴 sources 주입 (Phase 3-a)
      └─ Model slot: synthesis_model (Phase 9)
       │
       ▼  { token stream }
       │
       ▼
    [Hook] post_synthesis  (HTML strip 등)
       │
       ▼
    세션 기록 (user + assistant)
       │
       ▼
    Memory Compression (Phase 12)
      └─ 긴 세션은 오래된 턴을 LLM 요약으로 압축
       │
       ▼
    sources 이벤트 + done 이벤트
```

## 구성 요소 카탈로그

| 모듈 | 역할 | Phase |
|---|---|---|
| `agent/intent_classifier.py` | LLM 의도 분류 + TTL 캐시 | 5 |
| `agent/router.py` | intent + 키워드 라우팅 | 1a, 5 |
| `agent/orchestrator.py` | 전체 파이프라인 조율 | 1b~ |
| `agent/planner.py` | 질의→ExecutionPlan | 1c |
| `agent/verifier.py` | 컨텍스트 충분성 판정 | 1c |
| `agent/memory.py` | 대화 이력 요약 | 12 |
| `agent/hooks.py` | 파이프라인 lifecycle hook | 10 |
| `agent/personas/` | Researcher/Comparator/Analyst/Procedural/Clarifier | 6, 11 |
| `agent/persona_router.py` | intent→persona 매핑 | 6 |
| `agent/persona_loader.py` | YAML 기반 custom personas | 14 |
| `agent/skills/` | 도메인 YAML 지식 팩 | 7 |
| `agent/state.py` | 파일 기반 세션 영속화 | 1a |
| `agent/session.py` | 인메모리 세션 관리 | 1a |
| `agent/loop.py` | ReAct legacy AgentLoop | 2 |

## Config 프리셋

### `smart_mode: true` — 권장 기본
활성화: IntentClassifier + Clarifier + Personas + Planner + Verifier + Legacy fallback

특성:
- 의도 정확성 + 다단계 검색 + 근거 검증
- LLM 호출 수: 질의당 약 4~8회
- 응답 시간: 기본 대비 2~3배

### 세부 opt-in 플래그
- `skills_enabled` + `skills_dir` — 도메인 지식 팩
- `custom_personas_dir` — 사용자 정의 persona
- `parallel_steps` — plan의 모든 step이 ToolSpec.parallel_safe=True면 병렬 실행 (search/lookup/compare 등 read-only 도구)
- `persist_sessions` + `sessions_dir` — 세션 영속화
- `models.*_model` — 컴포넌트별 Ollama 모델 분리 (Phase 9)

## 이벤트 계약 (SSE)

모든 이벤트는 `data: {json}\n\n` 형식. 클라이언트는 다음 타입을 처리:

| 이벤트 | 언제 | 필드 |
|---|---|---|
| `route` | `/auto` 라우팅 직후 | `mode`, `intent?` |
| `clarification` | ambiguous → 역질문 | `questions[]`, `is_fallback` |
| `thought` | reasoning (stream_reasoning=True) | `content`, `iteration` |
| `action` | 도구 호출 직전 | `content`(tool), `input`, `iteration` |
| `observation` | 도구 결과 | `content` (truncated), `iteration` |
| `token` | synthesis 스트림 | `content` |
| `sources` | 최종 참조 문서 | `sources[]` |
| `done` | 종료 | `session_id?` |

## Never-Raise 계약

모든 LLM 호출 컴포넌트(Planner, Verifier, Clarifier, IntentClassifier, Compressor)는 **예외를 전파하지 않습니다**. 실패 시 안전한 기본값:

- Planner: 단일 search fallback plan
- Verifier: sufficient=True (중립)
- Clarifier: 일반적 fallback 질문
- IntentClassifier: ambiguous (Clarifier 트리거)
- Compressor: None (압축 스킵)

이 계약으로 LLM 장애가 agent 전체를 중단시키지 않습니다.

## 핵심 아키텍처 규약

### Drafting vs Publishing 분리 (HIGH-1 / HIGH-2)

Planner 경로의 합성은 **답변을 만드는 단계**와 **클라이언트에 발행하는 단계**가 엄격히 분리됩니다.

- 합성은 `_collect_synthesis()`로 답변을 문자열 버퍼에 수집합니다(token 이벤트를 yield하지 않음).
- 합성 완료 후 최종 답변만 **단일 `{"type":"token"}` 이벤트**로 발행합니다.
- 결과: SSE token 이벤트가 답변 1개당 정확히 1번만 발행되며, OpenAI 호환 어댑터의 `content_parts` 누적이 항상 단일 답변만 포함합니다.
- `if answer.strip():` 가드로 빈 답변은 `sources`/`token` 이벤트 없이 `done`만 발행합니다.

### Session User 메시지 우선 기록 (HIGH-3)

- user 메시지는 `planner.plan()` **호출 전**에 세션에 기록됩니다 — follow-up 질의의 plan이 history를 활용할 수 있도록.
- planner → legacy fallback 시 `_stream_agent_legacy(skip_user_message=True)`로 호출하여 user 메시지 이중 기록을 방지합니다.

### Raw vs Normalized Query (MED-1)

- `pre_query` hook으로 정규화된 `query`는 router/planner/synthesis로 전달됩니다.
- 세션 `add_message`에는 `raw_query`를 기록합니다 — 사용자 입력 원본과 downstream 컨텍스트를 분리.

### Force-Answer Thought 이벤트

`AgentLoop`가 `max_iterations`에 도달하면 강제 답변 생성 전에 다음 thought 이벤트를 발행합니다:

```
{"type": "thought", "content": "max_iterations(N) 도달 — 강제 답변 생성"}
```

Final Answer 마커는 한국어("최종 답변")와 영어("Final Answer") 모두 탐색하며, parser fallback은 마커가 없고 `<10자`이면 force_answer로 escalate합니다.

### Parallel Step Gate

`parallel_steps=true`일 때, orchestrator는 plan의 모든 step을 검사하여 `ToolSpec.parallel_safe=True`이고 2개 이상인 경우에만 `asyncio.gather`로 병렬 실행합니다. 이전에는 `search` 전용이었으나 이제 메타데이터 기반으로 일반화되었습니다. 이벤트는 plan 순서대로 emit되어 SSE 계약이 보존됩니다.

내장 도구 `parallel_safe` 플래그:
- `search` / `lookup` / `compare` — `True` (read-only, 호출 순서 독립)
- `evaluate` / `list_documents` — `False` (일관성/의존성 보호를 위해 직렬)

### 성능 최적화

- `AgentLoop._decompose_query`는 최대 50개 LRU 캐시를 보유합니다(동일 쿼리 재사용).
- `FileBackedSessionStore.cleanup_expired`는 파일 `mtime` 기반으로 판정합니다 — JSON 로드 없이 빠른 만료 체크.
- `_LazyIntentClassifier`는 `asyncio.Lock`으로 첫 인스턴스화 race를 방지합니다.
- 모든 LLM 호출 모듈에 `keep_alive` 파라미터가 전파됩니다(`rag.agent.ollama_keep_alive`로 제어, 기본 `"5m"`).

## 신규 config 필드 요약 (4차 리뷰)

| 필드 | 기본값 | 목적 |
|---|---|---|
| `observation_preview_limit` | `300` | observation 이벤트의 본문 미리보기 길이 제한(문자) |
| `ollama_keep_alive` | `"5m"` | 모든 LLM 호출의 Ollama `keep_alive`. 이전에는 `-1` 하드코딩 |

## 테스트 커버리지 (Phase 15b 완료 기준)

- 전체: **1097 passed**, 25 skipped
- 신규 테스트 (Phase 5~15b): **310개**
- 통합 테스트: `tests/test_rag_server_integration.py` (FastAPI TestClient)

## YAML 설정 예시

### smart_mode (권장)
```yaml
rag:
  agent:
    enabled: true
    smart_mode: true
```

### 개별 플래그 수동 제어
```yaml
rag:
  agent:
    enabled: true
    planner_enabled: true
    verifier_enabled: true
    intent_classifier_enabled: true
    clarifier_enabled: true
```

## 확장 지점

1. **새 persona 추가**: YAML에 정의하고 `custom_personas_dir`에 배치
2. **도메인 지식 추가**: YAML에 skill 정의, `skills_dir` 추가
3. **커스텀 Hook**: `orchestrator.register_hook(point, fn)` 호출
4. **새 모델 슬롯**: `AgentModelsConfig`에 필드 추가 + `_model_for(slot)` 대응

## 설계 원칙

1. **Opt-in 기본값**: 새 기능은 `*_enabled: false`로 시작, 기존 동작 무변경
2. **Never-raise 계약**: LLM 실패도 agent를 중단시키지 않음
3. **SSE 계약 보존**: 이벤트 타입 추가만, 기존 필드/순서 변경 금지
4. **테스트 우선**: 신규 모듈은 단위 + 통합 테스트 동시 작성
5. **설정 응집**: 모든 기능은 `rag.agent` config 아래 단일 네임스페이스
