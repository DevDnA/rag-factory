"""학습된 모델의 자동 평가 모듈입니다."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from .config import EvalConfig, SLMConfig

from .models import EvalResult, QAPair
from .utils import get_logger, ollama_generate, run_async, run_bounded

logger = get_logger("evaluator")


_bleu_metric = None
_rouge_metric = None
_korean_tokenizer = None
_korean_tokenizer_checked = False


def _load_bleu():
    global _bleu_metric
    if _bleu_metric is None:
        import evaluate

        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric


def _load_rouge():
    global _rouge_metric
    if _rouge_metric is None:
        import evaluate

        _rouge_metric = evaluate.load("rouge")
    return _rouge_metric


def _load_korean_tokenizer():
    """한국어 형태소 분석 토크나이저를 로드합니다 (kiwipiepy 필요).

    ``kiwipiepy``가 설치되어 있으면 형태소 분석 함수를 반환하고,
    설치되지 않았으면 ``None``을 반환합니다.

    반환값
    -------
    callable | None
        텍스트를 받아 형태소 리스트를 반환하는 함수, 또는 ``None``.
    """
    global _korean_tokenizer, _korean_tokenizer_checked
    if _korean_tokenizer_checked:
        return _korean_tokenizer
    _korean_tokenizer_checked = True
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()

        def tokenize(text: str) -> list[str]:
            return [token.form for token in kiwi.tokenize(text)]

        _korean_tokenizer = tokenize
        logger.info("한국어 형태소 분석 토크나이저 로드됨 (kiwipiepy)")
    except ImportError:
        logger.debug("kiwipiepy 미설치 — 기본 공백 토크나이저 사용")
        _korean_tokenizer = None
    return _korean_tokenizer


def _preprocess_for_metrics(text: str, korean_tokenizer=None) -> str:
    """메트릭 계산을 위해 텍스트를 전처리합니다.

    한국어 토크나이저가 제공되면 형태소 단위로 분리하여
    공백으로 연결합니다. 이를 통해 BLEU/ROUGE가 형태소 단위로
    계산되어 한국어 평가의 정확도가 향상됩니다.

    매개변수
    ----------
    text:
        전처리할 텍스트.
    korean_tokenizer:
        형태소 분석 함수. ``None``이면 원본 텍스트를 반환합니다.

    반환값
    -------
    str
        전처리된 텍스트.
    """
    if korean_tokenizer is not None:
        return " ".join(korean_tokenizer(text))
    return text


class RetrievalEvaluator:
    """qa.parquet의 retrieval_gt 기준으로 검색 품질을 평가합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.db_path = Path(config.paths.output) / config.rag.vector_db_path

    def evaluate(self, qa_path: Path, top_k: int = 5) -> dict[str, float]:
        """Qdrant 벡터 검색의 Recall@K, MRR, Hit@K를 계산합니다."""
        import pandas as pd

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

        qa_df = pd.read_parquet(qa_path)
        if "retrieval_gt" not in qa_df.columns:
            raise ValueError(f"{qa_path}에 retrieval_gt 컬럼이 없습니다")

        logger.info("검색 평가 시작 — %d개 QA, top_k=%d", len(qa_df), top_k)

        model = SentenceTransformer(self.config.rag.embedding_model)
        client = QdrantClient(path=str(self.db_path))
        collection = self.config.rag.collection_name

        hits_at_1 = 0
        hits_at_k = 0
        reciprocal_ranks: list[float] = []
        recall_at_k_list: list[float] = []
        total = 0

        for _, row in qa_df.iterrows():
            query = str(row["query"])
            gt_lists = row["retrieval_gt"]
            gt_ids: set[str] = set()
            for gt_group in gt_lists:
                if isinstance(gt_group, list):
                    gt_ids.update(gt_group)
                else:
                    gt_ids.add(str(gt_group))

            if not gt_ids:
                continue

            query_vec = model.encode(query, show_progress_bar=False).tolist()
            results = client.query_points(
                collection_name=collection,
                query=query_vec,
                limit=top_k,
                with_payload=True,
            ).points

            retrieved_ids = [p.payload.get("doc_id", "") for p in results if p.payload]

            if retrieved_ids and retrieved_ids[0] in gt_ids:
                hits_at_1 += 1

            if gt_ids & set(retrieved_ids):
                hits_at_k += 1

            rr = 0.0
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in gt_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            found = len(gt_ids & set(retrieved_ids))
            recall_at_k_list.append(found / len(gt_ids))

            total += 1

        client.close()

        if total == 0:
            logger.warning("평가할 QA 쌍이 없습니다")
            return {}

        metrics = {
            "hit@1": round(hits_at_1 / total, 4),
            f"hit@{top_k}": round(hits_at_k / total, 4),
            "mrr": round(sum(reciprocal_ranks) / total, 4),
            f"recall@{top_k}": round(sum(recall_at_k_list) / total, 4),
        }

        logger.info("검색 평가 완료 — %d건: %s", total, metrics)
        return metrics

    def print_summary(self, metrics: dict[str, float]) -> None:
        from rich.console import Console

        con = Console()
        if not metrics:
            con.print("[yellow]검색 평가 결과가 없습니다[/yellow]")
            return

        table = Table(title="검색 품질 평가 결과")
        table.add_column("메트릭", style="cyan")
        table.add_column("점수", justify="right", style="bold")
        table.add_column("설명")

        descriptions = {
            "hit@1": "Top-1에 정답 문서가 있는 비율",
            "mrr": "정답 문서의 평균 역순위 (Mean Reciprocal Rank)",
        }

        for key, value in metrics.items():
            if key.startswith("hit@"):
                k = key.split("@")[1]
                desc = descriptions.get(key, f"Top-{k}에 정답 문서가 있는 비율")
            elif key.startswith("recall@"):
                k = key.split("@")[1]
                desc = f"Top-{k}에서 정답 문서를 찾은 비율"
            else:
                desc = descriptions.get(key, "")
            table.add_row(key, f"{value:.4f}", desc)

        con.print(table)


class ModelEvaluator:
    """Ollama 모델을 QA 쌍으로 평가합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.eval_config: EvalConfig = config.eval
        self.api_base = config.teacher.api_base.rstrip("/")
        self.timeout = config.teacher.timeout
        self.max_concurrency = config.teacher.max_concurrency

    async def _generate(
        self, client: httpx.AsyncClient, model_name: str, question: str
    ) -> str:
        return await ollama_generate(
            client,
            self.api_base,
            model_name,
            question,
            self.timeout,
            max_tokens=self.eval_config.max_tokens,
        )

    def _compute_scores(self, reference: str, generated: str) -> dict[str, float]:
        """설정된 메트릭으로 점수를 계산합니다.

        한국어 프로젝트(``project.language == "ko"``)에서는 형태소 분석
        토크나이저로 텍스트를 전처리한 뒤 BLEU/ROUGE를 계산합니다.
        이를 통해 한국어의 교착어 특성이 반영된 정확한 점수를 얻습니다.
        """
        scores: dict[str, float] = {}
        metrics = self.eval_config.metrics

        # 한국어 프로젝트일 때 형태소 분석 토크나이저 사용
        tokenizer = None
        if self.config.project.language == "ko":
            tokenizer = _load_korean_tokenizer()

        ref = _preprocess_for_metrics(reference, tokenizer)
        gen = _preprocess_for_metrics(generated, tokenizer)

        if not gen.strip():
            return {
                m: 0.0
                for m in ["bleu", "rouge1", "rouge2", "rougeL"]
                if m in metrics or "rouge" in metrics
            }

        if "bleu" in metrics:
            bleu = _load_bleu()
            try:
                result = bleu.compute(
                    predictions=[gen],
                    references=[[ref]],
                )
                scores["bleu"] = round(result["bleu"], 4)
            except ZeroDivisionError:
                scores["bleu"] = 0.0

        if "rouge" in metrics:
            rouge = _load_rouge()
            try:
                result = rouge.compute(
                    predictions=[gen],
                    references=[ref],
                )
                scores["rouge1"] = round(result["rouge1"], 4)
                scores["rouge2"] = round(result["rouge2"], 4)
                scores["rougeL"] = round(result["rougeL"], 4)
            except ZeroDivisionError:
                scores["rouge1"] = 0.0
                scores["rouge2"] = 0.0
                scores["rougeL"] = 0.0

        return scores

    async def _llm_judge_score(
        self,
        client: httpx.AsyncClient,
        reference: str,
        generated: str,
        question: str,
    ) -> float:
        """Teacher LLM이 생성 답변의 의미적 정확성을 0~5 점수로 평가합니다."""
        prompt = (
            "다음 질문에 대한 참조 답변과 생성 답변을 비교하여 "
            "생성 답변의 품질을 0~5 점수로 평가하세요.\n\n"
            f"질문: {question}\n"
            f"참조 답변: {reference}\n"
            f"생성 답변: {generated}\n\n"
            "평가 기준:\n"
            "- 5: 참조 답변과 의미적으로 동일하며 정확함\n"
            "- 4: 핵심 정보가 모두 포함되어 있으나 표현이 다름\n"
            "- 3: 핵심 정보의 일부가 포함됨\n"
            "- 2: 관련 내용이지만 핵심 정보가 누락됨\n"
            "- 1: 부분적으로만 관련됨\n"
            "- 0: 전혀 관련 없거나 잘못된 답변\n\n"
            '반드시 JSON 형식으로만 응답: {"score": 정수}'
        )

        model = self.eval_config.llm_judge_model or self.config.teacher.model
        response = await ollama_generate(
            client,
            self.api_base,
            model,
            prompt,
            self.timeout,
            max_tokens=50,
            format="json",
            think=False,
        )

        import re

        try:
            data = json.loads(response)
            score = int(data.get("score", 0))
            return max(0, min(5, score)) / 5.0
        except (json.JSONDecodeError, ValueError, TypeError):
            match = re.search(r'"score"\s*:\s*(\d)', response)
            if match:
                return max(0, min(5, int(match.group(1)))) / 5.0
            return 0.0

    async def _batch_llm_judge(
        self,
        client: httpx.AsyncClient,
        items: list[tuple[str, str, str]],
    ) -> list[float]:
        qa_list = ""
        for i, (question, reference, generated) in enumerate(items):
            qa_list += (
                f"[{i}] 질문: {question[:100]}\n"
                f"    참조: {reference[:150]}\n"
                f"    생성: {generated[:150]}\n"
            )

        prompt = (
            f"다음 {len(items)}개의 생성 답변을 참조 답변과 비교하여 0~5점으로 평가하세요.\n"
            "5=의미적 동일, 3=핵심 포함, 0=전혀 무관\n\n"
            f"{qa_list}\n"
            '반드시 JSON: {{"scores": [{{"id": 0, "score": 점수}}, ...]}}'
        )

        model = self.eval_config.llm_judge_model or self.config.teacher.model
        api_base = self.config.teacher.api_base
        timeout = self.config.teacher.timeout

        response = await ollama_generate(
            client,
            api_base,
            model,
            prompt,
            timeout,
            max_tokens=200,
            format="json",
            think=False,
        )

        results = [0.0] * len(items)
        try:
            data = json.loads(response)
            scores_list = data.get("scores", data.get("items", []))
            if isinstance(data, list):
                scores_list = data
            for item in scores_list:
                idx = int(item.get("id", -1))
                score = int(item.get("score", 0))
                if 0 <= idx < len(items):
                    results[idx] = max(0, min(5, score)) / 5.0
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return results

    async def _evaluate_one(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        pair: QAPair,
    ) -> EvalResult:
        generated = await self._generate(client, model_name, pair.question)
        scores = self._compute_scores(pair.answer, generated)

        return EvalResult(
            question=pair.question,
            reference_answer=pair.answer,
            generated_answer=generated,
            scores=scores,
        )

    async def _evaluate_async(
        self,
        qa_pairs: list[QAPair],
        model_name: str,
    ) -> list[EvalResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: list[EvalResult] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        async with httpx.AsyncClient() as client:
            with progress:
                task_id = progress.add_task("모델 평가 중...", total=len(qa_pairs))

                tasks = [
                    run_bounded(
                        semaphore,
                        self._evaluate_one(client, model_name, pair),
                        progress,
                        task_id,
                    )
                    for pair in qa_pairs
                ]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)

            for item in gathered:
                if isinstance(item, BaseException):
                    logger.error("평가 실패: %s", item)
                    continue
                results.append(item)

            if "llm_judge" in self.eval_config.metrics and results:
                batch_size = 10
                for i in range(0, len(results), batch_size):
                    batch = results[i : i + batch_size]
                    items = [
                        (r.question, r.reference_answer, r.generated_answer)
                        for r in batch
                    ]
                    try:
                        judge_scores = await self._batch_llm_judge(client, items)
                        for j, score in enumerate(judge_scores):
                            batch[j].scores["llm_judge"] = score
                    except Exception as e:
                        logger.error("LLM Judge 배치 실패: %s", e)

        logger.info("평가 완료: %d/%d 성공", len(results), len(qa_pairs))
        return results

    def evaluate(self, qa_pairs: list[QAPair], model_name: str) -> list[EvalResult]:
        """QA 쌍으로 모델을 평가합니다."""
        if not qa_pairs:
            logger.warning("평가할 QA 쌍이 없습니다")
            return []

        samples = qa_pairs[: self.eval_config.max_samples]
        return run_async(self._evaluate_async(samples, model_name))

    def save_results(self, results: list[EvalResult], path: Path) -> None:
        """평가 결과를 JSON 파일로 저장합니다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in results]
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("평가 결과 저장: %s (%d건)", path, len(results))

    def check_quality_gate(
        self,
        results: list[EvalResult],
        thresholds: dict[str, float],
    ) -> tuple[bool, dict[str, float]]:
        """평가 결과가 품질 임계값을 충족하는지 확인합니다."""
        if not results:
            return False, {}

        averages: dict[str, float] = {}
        all_keys: list[str] = []
        for r in results:
            for k in r.scores:
                if k not in all_keys:
                    all_keys.append(k)

        for key in all_keys:
            vals = [r.scores[key] for r in results if key in r.scores]
            if vals:
                averages[key] = sum(vals) / len(vals)

        passed = True
        for metric, threshold in thresholds.items():
            avg = averages.get(metric, 0.0)
            if avg < threshold:
                logger.warning(
                    "품질 게이트 실패: %s 평균 %.4f < 임계값 %.4f",
                    metric,
                    avg,
                    threshold,
                )
                passed = False

        if passed:
            logger.info(
                "품질 게이트 통과: %s", {k: f"{v:.4f}" for k, v in averages.items()}
            )

        return passed, averages

    def print_summary(self, results: list[EvalResult]) -> None:
        """평가 결과 요약을 출력합니다."""
        from rich.console import Console

        con = Console()

        if not results:
            con.print("[yellow]평가 결과가 없습니다[/yellow]")
            return

        all_keys: list[str] = []
        for r in results:
            for k in r.scores:
                if k not in all_keys:
                    all_keys.append(k)

        table = Table(title="평가 결과 요약")
        table.add_column("메트릭", style="cyan")
        table.add_column("평균", justify="right", style="bold")
        table.add_column("최소", justify="right")
        table.add_column("최대", justify="right")

        for key in all_keys:
            vals = [r.scores[key] for r in results if key in r.scores]
            if vals:
                table.add_row(
                    key,
                    f"{sum(vals) / len(vals):.4f}",
                    f"{min(vals):.4f}",
                    f"{max(vals):.4f}",
                )

        con.print(table)
        con.print(f"\n총 {len(results)}건 평가 완료")
