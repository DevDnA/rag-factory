# 환경별 RAG 구성 가이드

slm-factory의 Agent RAG는 메모리·latency·품질 trade-off가 큽니다. 본 가이드는
실측 벤치(`FINDINGS.md`)를 바탕으로 사용자 환경에 맞는 구성을 제시합니다.

## 1. 메모리 예산 산정

| Ollama 모델 | VRAM | 비고 |
|---|---|---|
| qwen3.5:35b-a3b | 22GB | MoE 합성용 최고 품질, 대부분 환경 swap 발생 |
| gemma4:26b | 16GB | 고품질, 24GB에선 단독만 가능 |
| gemma3:12b | 9GB | thinking + 256K context |
| gemma4:e4b | 9GB | 8B edge 변종, 짧은 답변 경향 |
| qwen3.5:9b | 8GB | **주력 추천**, JSON·합성 모두 안정 |
| gemma4:e2b | 7GB | 5B edge 변종, 판정용으로 가능 |
| qwen3.5:4b | 5GB | 빠른 합성·판정 |
| qwen2.5:1.5b | 1GB | 너무 작아 합성 부적합 (실측: 자기계발 톤·디테일 누락) |

**+ 항상 5~7GB 추가 점유**: macOS 시스템, RAG 서버(임베딩+리랭커+FastAPI),
Python 런타임 등.

### 추천 동시 상주 조합

| 메모리 | 합성 | 판정 | 합 | 비고 |
|---|---|---|---|---|
| 16GB | qwen3.5:4b | qwen3.5:4b | 5GB | quality 단순 단일 |
| 24GB | qwen3.5:9b | qwen3.5:9b | 8GB | **벤치 검증 우승** ⭐ |
| 24GB | qwen3.5:4b | qwen3.5:9b | 13GB | latency 우선 |
| 24GB | gemma3:12b | qwen3.5:4b | 14GB | gemma 합성 시도 |
| 32GB+ | qwen3.5:35b-a3b | qwen3.5:9b | 30GB | 큰 합성기 분리 (가능) |
| 48GB+ | qwen3.5:35b-a3b | gemma3:12b | 31GB | 진정한 cross-family |

## 2. 사용 시나리오별 권장 프리셋

### A. 품질 최우선 (RFP·법률·정밀 답변)

```yaml
rag:
  ollama_model: "qwen3.5:9b"
  request_timeout: 600.0
  max_tokens: -1
  agent:
    enabled: true
    quality_mode: true             # cascade로 planner/verifier/clarifier ON
    # ralph_loop_enabled는 cascade로 자동 ON
    ralph_loop_max_iterations: 2
    ralph_loop_quality_threshold: 7.0
    ralph_loop_strategy: "reset"
    ralph_loop_state_dir: "ralph_state"
    intent_verbalization_enabled: true
    parallel_steps: false          # macOS Python 3.14 SIGSEGV 회피
    native_thinking: true
    ollama_keep_alive: "168h"
    models:
      synthesis_model: "qwen3.5:9b"
      reviewer_model:  "qwen3.5:9b"
      scorer_model:    "qwen3.5:9b"
      reflector_model: "qwen3.5:9b"
      clarifier_model: "qwen3.5:9b"
      planner_model:   "qwen3.5:9b"
      verifier_model:  "qwen3.5:9b"
      router_model:    "qwen3.5:9b"
```

**예상 동작**: query당 평균 130s, promise rate 67%, score 평균 8.0/10. 답변 길이
~2000자, 출처 4-5개 인용. **RFP 같은 도메인에 권장.**

### B. 균형 (일반 사용 — 채팅 응답성·답변 풍부도)

```yaml
rag:
  ollama_model: "qwen3.5:4b"
  request_timeout: 600.0
  max_tokens: -1
  agent:
    enabled: true
    quality_mode: true
    ralph_loop_enabled: false      # 품질 게이트 OFF로 latency 단축
    native_thinking: false
    parallel_steps: false
    ollama_keep_alive: "168h"
    intent_verbalization_enabled: true
    models:
      synthesis_model: "qwen3.5:4b"   # 빠른 합성
      planner_model:   "qwen3.5:4b"
      verifier_model:  "qwen3.5:4b"
      router_model:    "qwen3.5:4b"
      clarifier_model: "qwen3.5:4b"   # 9b 의존성 완전 제거
      reviewer_model:  "qwen3.5:9b"   # ralph OFF라 미호출
      scorer_model:    "qwen3.5:9b"   # 미호출
      reflector_model: "qwen3.5:9b"   # 미호출
```

**예상 동작**: query당 평균 ~55s, 답변 ~2000자(plan별 다중 검색 컨텍스트로 4b도
풍부한 답변 생성), 자체 게이트 평가 없음. **일반 채팅·문서 Q&A에 권장.**

### C. 최저 latency (실시간 응답 필요)

```yaml
rag:
  agent:
    enabled: false                  # /auto가 항상 simple RAG로
    # OR
    quality_mode: false
    planner_enabled: false
    verifier_enabled: false
    ralph_loop_enabled: false
```

또는 chat.html 등 클라이언트가 `/v1/chat/completions`를 `model="slm-factory-rag"`
(simple) 으로 직접 호출.

**예상 동작**: query당 ~12-18s, 단일 검색+합성. **품질 손실 큼** — 도메인 약어가
다의적인 RFP에선 첫 검색 청크 하나에 의존해 잘못된 추론 가능 (실측: NMS query를
"공급망 관리 시스템"으로 오해석한 사례 있음). 단순 fact 질의에만 권장.

## 3. 단계별 latency 단축 가이드

기본(품질 모드)에서 시작해 단계적으로 latency를 줄이고 싶을 때:

| 단계 | 변경 | 누적 latency (예시 query) | 품질 영향 |
|---|---|---|---|
| 시작 | 품질 모드 (구성 A) | 130s | promise 67% |
| 1 | `native_thinking: false` | 107s | 판정 품질 약간↓ |
| 2 | `parallel_steps: false` | 84s | 큰 차이 없음, 안정성↑ |
| 3 | `ralph_loop_enabled: false` | 74s | 자체 게이트 사라짐, 답변은 첫 합성 그대로 |
| 4 | `synthesis_model: "qwen3.5:4b"` | 55s | 답변 디테일 약간↓, 길이는 유지 |
| 5 | `planner/verifier/router: "qwen3.5:4b"` | 50s | plan 복잡도 따라 변동, 평균 -5s |
| 6 | `synthesis_model: "qwen2.5:1.5b"` | 21s | **답변 품질 절벽 — 권장 안 함** |
| 7 | simple RAG path (agent 우회) | 12-18s | 다단계 검색 사라짐, 도메인 부정확 |

**대부분 환경에서 4단계까지가 합리적 한계점**. 5단계 이상은 답변 품질 손실이
latency 이득을 정당화하기 어렵습니다.

## 4. 알려진 호환성 이슈

### macOS Python 3.14 + sentence-transformers
**증상**: 첫 query 처리 중 SIGSEGV 또는 SIGABRT (semaphore leak 메시지 동반)
**원인**: `loky` (joblib) 멀티프로세싱과 macOS Python 3.14 호환성
**회피**:
- `parallel_steps: false` (config)
- `TOKENIZERS_PARALLELISM=false` (env)
- `OMP_NUM_THREADS=1` (env)

### Ollama keep_alive
**증상**: orchestrator의 모든 LLM 호출이 400 Bad Request (`time: missing unit in
duration "-1"`)
**원인**: keep_alive 문자열 `"-1"`은 Go duration 파서가 단위 부족으로 거부.
정수 `-1`은 OK이나 YAML/Pydantic이 문자열로 직렬화함.
**회피**: `ollama_keep_alive: "168h"` (1주, 사실상 영구)

### 24GB 통합 메모리에서 큰 모델 swap
**증상**: query당 5~14분, "Batches: 0%" 로그가 멈춤
**원인**: 35b/26b가 다른 모델과 동시 로드되며 LRU eviction · swap thrashing
**회피**: 단일 모델 또는 9b+4b 동시 상주만 사용

## 5. 모델 cold start 회피 (LaunchAgent)

매 macOS 로그인 시 자동으로 모델을 메모리에 영구 핀하는 LaunchAgent:

```bash
# 스크립트
~/.local/bin/ollama-warmup-slm-factory.sh

# 등록
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.devdna.slm-factory.ollama-warmup.plist

# 즉시 실행
launchctl kickstart -p gui/$UID/com.devdna.slm-factory.ollama-warmup

# 로그 확인
cat /tmp/ollama-warmup-slm-factory.log
```

스크립트의 `MODEL` 변수를 변경하면 다른 모델을 핀할 수 있습니다.

## 6. 답변 quality 측정 방법

```bash
# 1) 기준선 측정 (현재 구성)
uv run python benchmark/bench.py --run-name baseline --threshold 7.0

# 2) 구성 변경 후 측정
# (my/project.yaml 수정 + 서버 재시작)
uv run python benchmark/bench.py --run-name variant --threshold 7.0

# 3) 비교
uv run python benchmark/analyze.py
```

`benchmark/queries.json`에 자신의 도메인 query를 추가하면 도메인별 평가 가능.
형식:
```json
{
  "queries": [
    {"id": "q1", "intent": "fact", "query": "..."},
    {"id": "q2", "intent": "compare", "query": "..."}
  ]
}
```

## 요약 결정 트리

```
시작
├── 응답 시간 < 30s 절대 필요?
│   └── 예 → simple RAG path (구성 C). 답변 품질 절벽 감수.
├── 도메인 정확도가 critical?
│   └── 예 → 품질 모드 (구성 A). promise 67% 보증.
├── 일반 채팅·검색 용도?
│   └── 예 → 균형 모드 (구성 B). ~55s, 답변 풍부, 게이트 없음.
└── VRAM > 48GB?
    └── 예 → cross-family 시도 (35b 합성 + 9b 판정). FINDINGS.md
            "후속 작업 후보" 참고.
```
