# RAG 벤치마크 자료 (historical)

> **이 디렉터리는 historical 자료만 보존합니다.**
> Ralph quality loop 시절 사용하던 `bench.py` / `analyze.py`는 제거되었습니다
> (관련 SSE 이벤트 `ralph_iteration`, `reflector_ok`, `reviewer_passed`, `promise`가
> 더 이상 발행되지 않아 코드 자체가 dead). 새 벤치 도구는 simple RAG vs Agent RAG
> 비교 목적으로 별도 재구성 예정.

## 보존 자료

| 파일 | 역할 |
|---|---|
| `queries.json` | RFP 도메인 평가셋 — intent 라벨 포함, 재구성 시 재사용 가능 |
| `results/` | 과거 run의 raw JSON (Ralph quality loop 시절 스키마 — 참조 전용) |
| `FINDINGS.md` | 메모리 한계, max_iterations 영향 등 발견사항 |
| `CONFIG_GUIDE.md` | 메모리·latency·품질 trade-off별 환경 권장 프리셋 |

## 환경 호환성 메모

- **macOS Python 3.14 + sentence-transformers**: `parallel_steps=true`이면
  `loky` (joblib) 멀티프로세싱이 SIGSEGV 유발. 회피: `parallel_steps: false`,
  `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` 환경변수 설정.
- **Ollama keep_alive**: 문자열 `"-1"`은 단위 누락으로 400 반환. 정수 `-1` 또는
  `"168h"` 같은 duration 문자열 사용.
- **24GB 통합 메모리**: 합성 35b/26b + 판정 분리는 swap thrashing. 단일 모델 또는
  9b+4b 분리가 한계.
