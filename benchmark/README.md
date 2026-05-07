# Ralph Quality Loop 벤치마크

oh-my-openagent의 ralph-loop을 슴-펙토리에 이식한 ``RAGQualityLoop``의
threshold 미세조정용 평가 하니스입니다.

## 구성

| 파일 | 역할 |
|---|---|
| `queries.json` | 도메인 특화 평가셋(intent 라벨 포함) |
| `bench.py` | `/auto` SSE 스트림을 소비해 iter/score/latency 메트릭 수집 |
| `analyze.py` | `results/*.json`을 가중 합성 점수로 비교 |
| `results/` | run당 JSON 결과 (per-query + summary) |

## 실행 흐름

1. **서버 기동** — `my/project.yaml`에 `quality_mode: true`로 설정한 뒤
   ```bash
   slf rag --no-chat   # 인덱스만 빌드(이미 빌드되어 있다면 skip)
   slf rag             # 서버 + 채팅 (chat은 브라우저, 무시 가능)
   ```
   기다려서 ``http://localhost:8000/health/ready``가 200 반환할 때까지.

2. **벤치 실행** (threshold별 반복) —
   ```bash
   # threshold=7.0
   sed -i '' 's/ralph_loop_quality_threshold: .*/ralph_loop_quality_threshold: 7.0/' my/project.yaml
   # 서버 재시작 후
   uv run python benchmark/bench.py --run-name t70 --threshold 7.0
   ```

3. **분석** —
   ```bash
   uv run python benchmark/analyze.py
   ```
   composite 점수가 가장 높은 threshold가 추천됩니다.

## 평가 지표

| 지표 | 의미 | 좋음의 방향 |
|---|---|---|
| `promise_rate` | 게이트 통과(`<promise>DONE</promise>` 발행) 비율 | 높을수록 좋음 |
| `avg_iterations` | 평균 Ralph 반복 횟수 | 낮을수록 빠름·저비용 |
| `avg_last_score` | scorer 평균 점수(1~10) | 7.5 근처면 적정 |
| `avg_elapsed_sec` | 평균 응답 시간 | 사용자 인내심에 따라 |

## 가중치 조정

Composite 공식:
```
composite = promise_rate * 100 * w_promise
          - avg_iterations * w_iter
          - avg_elapsed_sec * w_latency
```

기본 `w_promise=1.0, w_iter=5.0, w_latency=0.1` — 시간보다 통과율을 우선.
응답 속도가 critical하면 `--w-latency 0.5`로 키워 실행:
```bash
uv run python benchmark/analyze.py --w-latency 0.5
```
