# RAG 벤치 결과 종합

> **이 문서는 historical bench 기록입니다.** Ralph quality loop 기능이 제거되었습니다.
> 아래 데이터는 simple RAG vs Agent RAG(planner+verifier) 구성 선택의 참고 자료로 활용하세요.
> ralph/reviewer/scorer 관련 수치는 현재 코드베이스에 해당 기능이 없으므로 재현 불가합니다.

실측 데이터·발견 사항을 정리합니다.

- **벤치셋**: `benchmark/queries.json` (RFP 도메인 3 query — compare/explain/howto)
- **환경**: Apple Silicon 24GB 통합 메모리, Python 3.14, Ollama
- **데이터 위치**: `benchmark/results/*.json` (Ralph quality loop 시절 스키마, 참조 전용)

## Run 매트릭스

| run | 합성 | 판정 | strategy | iter | promise | score | latency | 평가 |
|---|---|---|---|---|---|---|---|---|
| `t75_9b` | 9b | 9b | continue | 3 | 0% | 6.23 | 258s | 베이스라인 (열화) |
| **`t70_reset_iter2`** ⭐ | 9b | 9b | reset | 2 | **67%** | **8.00** | **130s** | 9b 통일 (현재 권장) |
| `t70_e4b_9b` | gemma4:e4b | qwen3.5:9b | reset | 2 | 33% | 5.50 | 108s | cross-family 시도 1 (실패) |
| `t70_9b_e2b` | qwen3.5:9b | gemma4:e2b | reset | 2 | — | — | crash | 메모리 압박 SIGSEGV |
| `t70_gemma3_12b_qwen9b` | gemma3:12b | qwen3.5:9b | reset | 2 | 0% | 6.50 | 158s | cross-family 시도 2 (실패) |

각 run의 raw 데이터(질의별 iteration·점수·통과 사유)는 동명 JSON에 저장.

## 결과 — 9b 통일 vs 다른 모든 시도

| 지표 | 9b 통일 (t70_reset_iter2) | 다른 시도들 |
|---|---|---|
| promise rate | 67% (3 query 중 2개 자체 게이트 통과) | 0~33% |
| avg score | 8.0 | 5.5~6.5 |
| avg latency | 130s | 108~258s |

**모든 비교 지표에서 9b 통일이 우월**합니다. 이론적으로 매력적이던 cross-family
하이브리드(다른 패밀리 합성+판정으로 self-evaluation bias 감소)는 두 번 실측에서
모두 실패. 메모리·합성 능력·판정 일관성을 종합하면 단일 9b가 현재 환경 최적.

## 핵심 발견

### 1. Threshold는 작은 레버, Reviewer가 진짜 게이트

베이스라인 t75_9b의 9개 iteration 중 8개가 **reviewer fail**로 종료.
점수가 9.0이어도 reviewer가 거부하면 promise 미발행.

```
[q1_compare] iter#3 score=9.0  reviewer=FAIL [completeness, hallucination]
             iter#4 score=6.0  reviewer=FAIL [grounding]
             iter#5 score=8.5  reviewer=FAIL [hallucination]
```

매 반복마다 다른 reviewer가 다른 사유로 reject — 9b reviewer의 비결정성이
큼. threshold 조정(7.0/7.5/8.0)은 score 게이트에만 영향, reviewer는 무관.

### 2. continue 전략은 답변을 악화시킬 수 있다

`continue`는 이전 반복의 reviewer 피드백을 누적해서 합성 프롬프트에 주입.
9b judge가 매번 다른 사유로 reject하면 누적 피드백이 **모순적 지시문**이 되어
합성기를 혼란시킴.

```
[q2_explain] iter#4 score=9.0
             iter#5 score=7.0  ← 누적 피드백 노이즈로 열화 시작
             iter#6 score=4.2  ← 더 악화
```

`reset` 전략(가장 최근 피드백만 사용)이 안정적.

### 3. 마지막 답변 ≠ 최선 답변

마지막 반복의 답변이 열화되면 사용자에게 낮은 품질 답변이 노출될 수 있다. 매
반복을 ranking하여 **최선 답변을 보존**하는 것이 중요하다.

### 4. 24GB 메모리에서 모델 분리 거의 불가

| 조합 | 메모리 합 | 결과 |
|---|---|---|
| qwen3.5:35b-a3b + qwen3.5:9b | 31GB | ❌ swap thrashing (5~14분/query) |
| qwen3.5:35b-a3b + qwen3.5:4b | 27GB | ❌ 4b evicted |
| gemma4:26b + qwen3.5:4b | 21GB | ❌ 4b evicted |
| gemma3:12b + qwen3.5:4b | 14GB | ✓ |
| gemma3:12b + qwen3.5:9b | 17GB | ✓ |
| gemma4:e4b + qwen3.5:9b | 17GB | ✓ |
| qwen3.5:9b 단독 | 8GB | ✓ |
| qwen3.5:4b + qwen3.5:9b | 13GB | ✓ |

24GB 환경에서 합성·판정 분리는 가능하지만 합성을 26b/35b로 키우는 순간
swap 발생. **단일 9b 또는 4b+9b 분리가 안전 한계**.

### 5. Cross-family LLM-as-judge 가설 — 이 환경에서 무효

이론: 합성자와 판정자가 다른 패밀리면 self-evaluation bias 감소 → reviewer가
합성자 결함을 더 잘 잡음. 두 번 측정 결과 모두 **반대로 작동**:

| 합성 | 판정 | promise | 비고 |
|---|---|---|---|
| qwen3.5:9b (단일) | qwen3.5:9b | 67% ⭐ | 단일이 우승 |
| gemma4:e4b | qwen3.5:9b | 33% | 답변 짧음 → reviewer reject |
| gemma3:12b | qwen3.5:9b | 0% | 답변 짧음 → reviewer reject |

**원인**: gemma 합성기는 답변을 압축적으로 작성(800~1500자 vs qwen 9b의 2000자+).
qwen reviewer는 짧은 답변을 "completeness/hallucination 부족"으로 일관 거부.
재시도해도 gemma 스타일이 일관돼 답변이 비슷한 길이로 다시 만들어지고 또 reject.

### 6. 추론 시간 단축 단계별 효과 (NMS 단일 query 기준)

| 변경 | latency | 메모 |
|---|---|---|
| 베이스라인 (max_iter=2, native_thinking=true) | 130s | 9b 합성, 67% promise |
| native_thinking=false | 107s | 판정 thinking 비활성, ~18% 단축 |
| parallel_steps=false | 74s | planner+verify+synth만 |
| max_tokens 4096→-1 | 84s | 답변 자연 종료(2382자) |
| **synthesis_model=qwen3.5:4b** | **63s** | **4b 합성, 답변 2306자 — 균형점** ⭐ |
| synthesis+planner+verifier+router=4b | 59s | 추가 -4s, plan 복잡도 변동성에 묻힘 |
| **synthesis=qwen2.5:1.5b** | **21s** | 답변 667자, 자기계발 톤·오타·디테일 누락 ❌ 실용 X |
| synthesis=gemma4:e4b | 36s | 답변 826자, 정확하나 일반론적, 출처 모호 |
| synthesis=gemma3:12b | 51s | 답변 825자, 4×4 MIMO 등 기술 사양은 정확하나 답변 짧음 |

**Latency vs 답변 풍부도 trade-off**: 합성기 크기에 답변 디테일이 비례. 1.5b는
실용 한계 미달, 4b가 균형점.

## NMS 도메인 query 답변 품질 비교

같은 query "저는 NMS 개발자입니다. 무엇을 개발해야하나요?"에 대해 (초기 측정):

| 모델 | latency | 길이 | 디테일 |
|---|---|---|---|
| **qwen3.5:9b** | 57s | 1344자 | 4 섹션 + 우선순위. BIS/24h SLA/통계. **MIMO·Preamble·60km/h 누락** |
| **qwen3.5:4b** | 63s | 2306자 | 5 섹션. **MIMO·Preamble·BIS·공인인증서·ETRI/TTA 모두 포함** ⭐ |
| gemma3:12b | 51s | 825자 | 짧지만 MIMO·Preamble·60km/h 정확 인용. 답변 가독성 약함 |
| gemma4:e4b | 36s | 826자 | 정확하지만 일반론적, 출처 모호 |
| qwen2.5:1.5b | 21s | 667자 | 자기계발 톤, 오타("공공과이파이"), 디테일 다수 누락 |

흥미: **qwen3.5:4b가 9b보다 RFP 기술 디테일을 더 풍부하게** 합성한 케이스도 있음.
plan에 따른 다중 검색 컨텍스트가 합성 모델 크기보다 답변 정확도에 더 중요.

### corpus profile + 라우팅 도입 후 재측정 (2026-05-08)

같은 도메인 query "NMS 개발업체가 준비해야할 것은"에 corpus_profile 자동 생성·
HyDE+Multi-Query+RRF·8 카테고리 IntentClassifier가 적용된 환경에서 3종 구성 비교:

| 구성 | 라우팅 | 데이터 소스 | latency | 길이 | 메모리 | 디테일 |
|---|---|---|---|---|---|---|
| **4b 통일** | `exploratory` ✓ | corpus | 49s | 1265자 | 5.8GB | BIS·ETRI·TTA·MIMO·60km/h·AP 제조사 3개 이상. 3 관점 |
| **9b 통일** | `ambiguous` ✗ → general | **웹만** | 25s | 561자 | 8.6GB | 일반 NMS 정의 (SNMP·SaaS). **RFP 무관** |
| **하이브리드** ⭐ | `exploratory` ✓ | corpus | 85s | 1400자 | 14.4GB | 4b 통일 + WPA3·OWE·IEEE 802.1x·ARP/DNS 스푸핑·SLA 2시간 통보 + 24시간 조치·국산제품 활용 계획서 |

(하이브리드 = synthesis_model: qwen3.5:9b, 나머지 모든 슬롯: qwen3.5:4b)

### 핵심 발견 (재측정)

**1. 큰 모델이 라우팅을 망친다** — 9b 통일은 "NMS"를 일반 IT 약어로 인식해
``ambiguous``로 분류, corpus 검색을 우회하고 웹 결과만으로 답변. **9b의 자기 학습
지식이 너무 풍부해서 corpus profile 컨텍스트를 무시**. 4b는 corpus profile에 더
의존해 정확한 분류함.

**2. 하이브리드(9b 합성 + 4b 라우팅)가 정성 최우수** — 4b 합성기가 잡은 도메인
디테일을 그대로 유지하면서 9b 합성기가 보안 표준(WPA3 Enterprise·OWE·IEEE
802.1x), 침입 방지(ARP/DNS 스푸핑·백도어 탐지), SLA 시간 표기(2시간 통보+24시간
조치), 행정 절차(국산제품 활용 계획서)까지 추가로 포섭.

**3. 트레이드오프 정량화**:
| 차원 | 4b 통일 → 하이브리드 |
|---|---|
| 답변 풍부도 | +10% (디테일 4개 신규 포섭) |
| latency | +73% (49s → 85s) |
| 메모리 | +150% (5.8GB → 14.4GB) |
| 24GB 환경 안정성 | 매우 안정 → borderline |

**4. 라우팅 모델은 작아야 한다** — corpus profile 기반 분류는 LLM 능력보다
*프롬프트 준수도*가 중요. 큰 모델은 자기 지식을 끌어와 분류를 흐림. 4b가 router
역할에 더 적합한 비직관적 결과.

## 권장 구성 (현재 데이터 기반)

> **참고**: ralph/quality_mode 관련 설정은 현재 제거되었습니다. 아래 현재 권장 구성을 사용하세요.

### 품질 우선 — 하이브리드 (9b 합성 + 4b 라우팅)
```yaml
rag:
  ollama_model: "qwen3.5:4b"
  max_tokens: -1
  hyde_enabled: true
  multi_query_enabled: true
  multi_query_count: 3
  corpus_profile:
    enabled: true
    auto_generate: true
  agent:
    enabled: true
    smart_mode: true
    parallel_steps: false
    ollama_keep_alive: "168h"
    models:
      synthesis_model: "qwen3.5:9b"
      clarifier_model: "qwen3.5:4b"
      planner_model:   "qwen3.5:4b"
      verifier_model:  "qwen3.5:4b"
      router_model:    "qwen3.5:4b"
```
**예상**: ~85s/query, 1400자, RFP 디테일 풍부

### 균형 (latency 우선) — 4b 통일
```yaml
rag:
  ollama_model: "qwen3.5:4b"
  max_tokens: -1
  agent:
    enabled: true
    smart_mode: true
    parallel_steps: false
    ollama_keep_alive: "168h"
    models:
      synthesis_model: "qwen3.5:4b"
      planner_model:   "qwen3.5:4b"
      verifier_model:  "qwen3.5:4b"
      router_model:    "qwen3.5:4b"
      clarifier_model: "qwen3.5:4b"
```
**예상**: avg ~55s/query, 답변 풍부 (2000자급)

## 재현 절차

> Ralph quality loop 시절 사용하던 `bench.py` / `analyze.py`는 제거됨.
> 위 매트릭스의 promise/score 수치는 현재 코드베이스에서 재현 불가하며 — latency / iter / strategy 영향 같은 정성적 관찰만 참고 가치 있음.
> 새 벤치 도구가 필요하면 simple RAG vs Agent RAG 비교용으로 신규 작성하고, 평가셋은 `benchmark/queries.json`을 재사용.

## 환경 호환성 메모

- **macOS Python 3.14 + sentence-transformers**: `parallel_steps=true`이면
  `loky` (joblib) 멀티프로세싱이 SIGSEGV 유발. 회피: `parallel_steps: false`,
  `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` 환경변수 설정.
- **Ollama keep_alive**: 문자열 `"-1"`은 단위 누락으로 400 반환. 정수 `-1` 또는
  `"168h"` 같은 duration 문자열 사용. 본 프로젝트는 `"168h"` 채택.
- **24GB 통합 메모리**: 합성 35b/26b + 판정 분리는 swap thrashing. 단일 모델 또는
  9b+4b 분리가 한계.

## 후속 작업 후보

- 더 큰 VRAM(48GB+) 환경에서 35b 합성 + 9b 라우팅 검증
- 답변 캐싱: 동일 query 재실행 시 즉시 반환 → UX 추가 단축
- 첫-토큰 단축: planner 진행 표시와 합성 시작을 병행 스트리밍
