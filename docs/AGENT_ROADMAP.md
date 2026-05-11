# Agent RAG 고도화 로드맵 (Phase 5~15)

oh-my-openagent (OMO)의 디자인 패턴을 RAG Q&A 컨텍스트에 이식하여 "똑똑한" agent를 구축하는 작업 문서.

## 개발 원칙

- **Opt-in 전부**: 새 기능은 모두 `*_enabled: false` 기본값. 기존 동작 무변경.
- **Never-raise 계약**: 모든 신규 LLM 호출 모듈은 실패 시 안전한 fallback 반환.
- **SSE 이벤트 계약 보존**: 신규 이벤트 타입은 추가만, 기존 필드/순서 변경 금지.
- **각 Phase 독립 PR 단위**: 완료 후 테스트 통과 + 회귀 0 확인 → 다음 Phase.

---

## Phase 5 — IntentGate (LLM 기반 의도 분류) [P0]

**목표**: 키워드 휴리스틱 대신 LLM으로 질의 의도를 분류.

**의도 카테고리**:
- `factual`: 단일 사실 (예: "제15조는?")
- `comparative`: 비교/대조 (예: "A와 B 차이?")
- `analytical`: 분석/종합 (예: "왜 이렇게 변경됐나?")
- `procedural`: 절차/방법 (예: "어떻게 신청하나?")
- `exploratory`: 탐색/개요 (예: "약관에 뭐가 있나?")
- `ambiguous`: 불명확 (예: "그거 어떻게 돼요?")

**산출물**:
- `src/slm_factory/rag/agent/intent_classifier.py` — `IntentClassifier`, `IntentDecision`
- `src/slm_factory/rag/agent/prompts.py` — `INTENT_CLASSIFIER_PROMPT`
- config: `intent_classifier_enabled`, `intent_classifier_cache_ttl`
- `tests/test_agent_intent_classifier.py`

**완료 조건**: `QueryRouter`가 `IntentClassifier` 활성 시 LLM 분류를 사용, 실패 시 키워드 fallback.

---

## Phase 6 — 전문 Agent Persona 시스템 [P0]

**목표**: 의도별 전용 agent persona + synthesis 스타일 + 도구 권한.

**Persona 정의**:
| Persona | Intent | Tools | Synthesis Style |
|---|---|---|---|
| Researcher | factual | search, lookup | 조항·수치·날짜 인용 중심 |
| Comparator | comparative | compare, search×2 | 표 형식 + 차이점 강조 |
| Analyst | analytical | search 반복 호출 | 다각도 종합 + 통찰 |
| Procedural | procedural | search, lookup | 단계별 번호 매김 |
| Clarifier | ambiguous | (none) | 역질문 1~2개 (Phase 11) |

**산출물**:
- `src/slm_factory/rag/agent/personas/` (디렉터리)
  - `base.py`, `researcher.py`, `comparator.py`, `analyst.py`, `procedural.py`
- `src/slm_factory/rag/agent/persona_router.py`
- prompts.py — persona별 synthesis prompt
- config: `personas_enabled`
- `tests/test_agent_personas.py`

**완료 조건**: orchestrator가 intent → persona → 전용 synthesis prompt 사용.

---

## Phase 7 — Skills 시스템 (도메인 지식 팩) [P1]

**목표**: 도메인별 지식을 YAML로 패킹하여 코드 수정 없이 추가 가능.

**디렉터리 구조**:
```
skills/
  legal/skill.yaml      # 법률 도메인
  lte-network/skill.yaml
  finance/skill.yaml
```

**Skill 스키마**:
- `name`, `description`
- `triggers`: 자동 활성 키워드/regex 패턴
- `prompt_addon`: synthesis prompt 주입 텍스트
- `validators`: 답변 형식 검증 규칙

**산출물**:
- `src/slm_factory/rag/agent/skills/` (로더)
- `skills/` (디렉터리, 예시 skill 2~3개)
- config: `skills_enabled`, `skills_dir`
- `tests/test_agent_skills.py`

---

## Phase 9 — Multi-Model Routing [P1]

**목표**: 작업 유형별 Ollama 모델 분리 (사용자 구성 가능).

**모델 슬롯**:
- `router_model`: intent 분류 (빠른 모델)
- `planner_model`: plan JSON 생성
- `synthesis_model`: 최종 답변 (가장 큰 모델)
- `verifier_model`: 검증 (가벼운 모델)
- `clarifier_model`: 명확화 질문 (중급 모델)

**산출물**:
- config: `rag.agent.models.*` nested struct
- 각 모듈이 모델 슬롯 참조 (기본값: 기존 `rag.ollama_model`)
- `tests/test_agent_model_routing.py`

---

## Phase 10 — Hooks 시스템 [P1]

**목표**: 파이프라인 주요 지점에 pre/post hook. 사용자가 커스터마이즈 가능.

**Hook points**:
- `pre_query`: query 정규화·확장
- `post_route`: 라우팅 결정 로깅
- `post_search`: source dedup·score boost
- `pre_synthesis`: skills prompt 적용
- `post_synthesis`: 인용 추출·포맷 검증
- `on_error`: graceful degradation

**산출물**:
- `src/slm_factory/rag/agent/hooks.py` — `HookRegistry`, built-in hooks
- orchestrator 통합
- config: `hooks_enabled`, `hook_modules` (사용자 hook 모듈 경로)
- `tests/test_agent_hooks.py`

---

## Phase 11 — Clarifier (Ambiguity 역질문) [P0]

**목표**: IntentGate가 `ambiguous` 분류 시 명확화 질문 1~2개 생성.

**산출물**:
- `src/slm_factory/rag/agent/personas/clarifier.py`
- `CLARIFIER_PROMPT`
- 신규 SSE 이벤트 `{type: "clarification", questions: [...]}`
- orchestrator가 `ambiguous` 의도 → Clarifier persona 실행
- config: `clarifier_enabled`, `clarifier_max_questions`
- `tests/test_agent_clarifier.py`

**주**: Phase 6과 동시 진행 가능.

---

## Phase 12 — Conversation Memory Compression [P1]

**목표**: 10턴 초과 시 가장 오래된 턴을 LLM 요약으로 압축.

**산출물**:
- `src/slm_factory/rag/agent/memory.py` — `ConversationCompressor`
- `SessionManager` / `FileBackedSessionStore`에 `compress_old_turns()` 메서드
- `MEMORY_COMPRESSION_PROMPT`
- config: `memory_compression_enabled`, `compress_after_turns`, `compress_target_chars`
- `tests/test_agent_memory.py`

---

## Phase 14 — 사용자 정의 Persona (YAML) [P2]

**목표**: YAML로 커스텀 agent persona 등록.

**산출물**:
- `src/slm_factory/rag/agent/persona_loader.py`
- config: `rag.agent.custom_personas` (list)
- `tests/test_agent_custom_personas.py`

---

## Phase 15 — 통합 Master Orchestrator [P0]

**목표**: 위 Phase들을 단일 파이프라인으로 통합. OMO의 Sisyphus 패턴.

```
사용자 질의
  → IntentGate (Phase 5)
  → Persona 선택 (Phase 6) / Clarifier (Phase 11)
  → Skills 자동 로드 (Phase 7)
  → 모델 슬롯 선택 (Phase 9)
  → Plan (Phase 1c) → Execute (Phase 3b) → Verify (Phase 1c) → Synthesize
  → Memory Compression (Phase 12)
  → 답변 + 메타데이터
```

**산출물**:
- `src/slm_factory/rag/agent/orchestrator.py` — 전체 파이프라인 통합
- 새 config: `rag.agent.smart_mode` (one-click 모든 P0 활성화 프리셋)
- 통합 테스트 `tests/test_rag_server_integration.py` 확장
- `docs/AGENT_ARCHITECTURE.md` — 최종 아키텍처 문서

**완료 조건**: `smart_mode: true` 설정 시 모든 P0 Phase가 조율되어 작동, 통합 테스트 통과.

---

## 진행 순서

1. Phase 5 (IntentGate)
2. Phase 11 (Clarifier) — Phase 6 기반이 되는 persona 뼈대와 함께
3. Phase 6 (Personas)
4. Phase 15 (부분 통합 — P0 연결)
5. Phase 7 (Skills)
6. Phase 10 (Hooks)
7. Phase 9 (Multi-Model)
8. Phase 12 (Memory)
9. Phase 14 (Custom Personas)
10. Phase 15 (최종 통합)

각 Phase 완료 시 체크박스 업데이트.

## 진행 체크리스트

- [x] Phase 5 — IntentGate
- [x] Phase 11 — Clarifier (persona 뼈대)
- [x] Phase 6 — Personas (Researcher, Comparator, Analyst, Procedural)
- [x] Phase 15a — P0 통합 (Phase 5+6+11) — `smart_mode` 프리셋
- [x] Phase 7 — Skills 시스템
- [x] Phase 10 — Hooks 시스템
- [x] Phase 9 — Multi-Model Routing
- [x] Phase 12 — Memory Compression
- [x] Phase 14 — Custom Personas
- [x] Phase 15b — 최종 통합 + 아키텍처 문서 — `docs/AGENT_ARCHITECTURE.md`
