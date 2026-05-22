# RAG 벤치마크

## 활성 평가셋: allganize/RAG-Evaluation-Dataset-KO

`queries.json`은 한국어 RAG 평가 표준 벤치 `allganize/RAG-Evaluation-Dataset-KO`(MIT) 의
**public + finance + law 3개 도메인 = 180 QA**로 구성됩니다.

### 핵심 메타 필드 (rag-factory 확장)

| 필드 | 용도 |
|---|---|
| `expected` | target_answer — substring 매칭으로 정답률 측정 |
| `expected_doc` | target_file_name — **retrieval ground-truth**, Recall@k·MRR 측정 가능 |
| `expected_page` | target_page_no — 페이지 단위 retrieval 평가 |
| `context_type` | paragraph / table / image — modality별 정확도 분해 |
| `domain` | finance / public / law — 도메인 일반화 측정 |
| `difficulty` | easy / medium / hard (target_answer 길이 휴리스틱) |

`intent`는 모두 `"qa"` placeholder — allganize 데이터셋에 intent 라벨이 없어 단일로 통일.
실제 라우팅은 `IntentClassifier`가 query별로 자동 분류함. `bench_streaming.py`는
`id`/`intent`/`query` 3개 필드만 사용하므로 위 메타 필드들은 무시되어도 안전.

## 평가셋 재생성 / 도메인 변경

```bash
# 기본: public + finance + law 180 QA
uv run python scripts/build_allganize_eval.py \
    --domains public finance law \
    --out benchmark/queries.json

# 전체 5개 도메인 300 QA
uv run python scripts/build_allganize_eval.py \
    --domains public finance law medical commerce \
    --out benchmark/queries.json
```

## 원본 PDF 다운로드 (corpus 인덱싱용)

allganize 데이터셋은 PDF를 번들하지 않고 `documents.csv`에 URL만 제공. 다수가
정부 viewer 페이지(`view.do?nttId=...`)라 자동 다운로드는 best-effort:

```bash
# best-effort 자동 다운로드 — 직링크 PDF는 곧장, viewer 페이지는 anchor 탐색 시도
uv run python scripts/fetch_allganize_pdfs.py \
    --domains public finance law \
    --out-dir allganize-eval-project/documents
```

실패한 PDF는 `allganize-eval-project/manual_download_list.txt`에 `(file_name, url)`로
기록됩니다. 해당 URL에서 직접 다운로드 후 같은 디렉토리에 **원본 file_name 그대로**
저장하면 `expected_doc` 매칭이 정상 동작합니다. PDF 재배포는 금지 — 로컬 다운로드만.

## 인덱싱 + 서빙

```bash
cd allganize-eval-project
rf rag                  # parse + index + serve
# 별도 터미널:
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
    uv run python benchmark/bench_streaming.py \
        --label allganize_baseline \
        --out benchmark/results/streaming_allganize_baseline.json
```

## Archive

| 파일 | 비고 |
|---|---|
| `queries-rfp-archive.json` | 이전 RFP(버스 공공와이파이) 47 QA 평가셋 — 참조 전용 |
| `results/` | Ralph quality loop 시절 raw JSON — 참조 전용 |
| `FINDINGS.md` | 메모리 한계, max_iterations 영향 등 과거 발견사항 |
| `CONFIG_GUIDE.md` | 메모리·latency·품질 trade-off별 환경 권장 프리셋 |

## 환경 호환성 메모

- **macOS Python 3.14 + sentence-transformers**: `parallel_steps=true`이면
  `loky` (joblib) 멀티프로세싱이 SIGSEGV 유발. 회피: `parallel_steps: false`,
  `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` 환경변수 설정.
- **Ollama keep_alive**: 문자열 `"-1"`은 단위 누락으로 400 반환. 정수 `-1` 또는
  `"168h"` 같은 duration 문자열 사용.
- **24GB 통합 메모리**: 합성 35b/26b + 판정 분리는 swap thrashing. 단일 모델 또는
  9b+4b 분리가 한계.
