"""Base 모델과 Fine-tuned 모델의 답변을 비교하는 모듈입니다."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

if TYPE_CHECKING:
    from .config import CompareConfig, SLMConfig

from .evaluator import _load_bleu, _load_korean_tokenizer, _load_rouge, _preprocess_for_metrics
from .models import CompareResult, QAPair
from .utils import get_logger, ollama_generate, run_async, run_bounded

logger = get_logger("comparator")


class ModelComparator:
    """QA 쌍으로 Base 모델과 Fine-tuned 모델을 비교합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.compare_config: CompareConfig = config.compare
        self.api_base = config.teacher.api_base.rstrip("/")
        self.timeout = config.teacher.timeout
        self.max_concurrency = config.teacher.max_concurrency

    async def _generate(self, client: httpx.AsyncClient, model_name: str, question: str) -> str:
        return await ollama_generate(
            client, self.api_base, model_name, question, self.timeout,
            max_tokens=self.compare_config.max_tokens,
        )

    def _compute_scores(self, reference: str, base_answer: str, finetuned_answer: str) -> dict[str, float]:
        """Base 모델과 Fine-tuned 모델 답변의 점수를 계산합니다.

        한국어 프로젝트에서는 형태소 분석 토크나이저로 텍스트를
        전처리한 뒤 점수를 계산합니다.
        """
        scores: dict[str, float] = {}
        metrics = self.compare_config.metrics

        # 한국어 프로젝트일 때 형태소 분석 토크나이저 사용
        tokenizer = None
        if self.config.project.language == "ko":
            tokenizer = _load_korean_tokenizer()

        ref = _preprocess_for_metrics(reference, tokenizer)
        base = _preprocess_for_metrics(base_answer, tokenizer)
        ft = _preprocess_for_metrics(finetuned_answer, tokenizer)

        if "bleu" in metrics:
            bleu = _load_bleu()
            base_result = bleu.compute(
                predictions=[base], references=[[ref]],
            )
            finetuned_result = bleu.compute(
                predictions=[ft], references=[[ref]],
            )
            scores["base_bleu"] = round(base_result["bleu"], 4)
            scores["finetuned_bleu"] = round(finetuned_result["bleu"], 4)

        if "rouge" in metrics:
            rouge = _load_rouge()
            base_result = rouge.compute(
                predictions=[base], references=[ref],
            )
            finetuned_result = rouge.compute(
                predictions=[ft], references=[ref],
            )
            scores["base_rouge1"] = round(base_result["rouge1"], 4)
            scores["base_rouge2"] = round(base_result["rouge2"], 4)
            scores["base_rougeL"] = round(base_result["rougeL"], 4)
            scores["finetuned_rouge1"] = round(finetuned_result["rouge1"], 4)
            scores["finetuned_rouge2"] = round(finetuned_result["rouge2"], 4)
            scores["finetuned_rougeL"] = round(finetuned_result["rougeL"], 4)

        return scores

    async def _compare_one(
        self,
        client: httpx.AsyncClient,
        pair: QAPair,
    ) -> CompareResult:
        """단일 QA 쌍에 대해 두 모델의 답변을 비교합니다."""
        base_model = self.compare_config.base_model
        finetuned_model = self.compare_config.finetuned_model

        base_answer, finetuned_answer = await asyncio.gather(
            self._generate(client, base_model, pair.question),
            self._generate(client, finetuned_model, pair.question),
        )
        scores = self._compute_scores(pair.answer, base_answer, finetuned_answer)

        return CompareResult(
            question=pair.question,
            reference_answer=pair.answer,
            base_answer=base_answer,
            finetuned_answer=finetuned_answer,
            scores=scores,
        )

    async def _compare_async(
        self, qa_pairs: list[QAPair],
    ) -> list[CompareResult]:
        """비동기로 전체 QA 쌍을 비교합니다."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: list[CompareResult] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        async with httpx.AsyncClient() as client:
            with progress:
                task_id = progress.add_task("모델 비교 중...", total=len(qa_pairs))

                tasks = [
                    run_bounded(semaphore, self._compare_one(client, pair), progress, task_id)
                    for pair in qa_pairs
                ]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for item in gathered:
            if isinstance(item, BaseException):
                logger.error("비교 실패: %s", item)
                continue
            results.append(item)

        logger.info("비교 완료: %d/%d 성공", len(results), len(qa_pairs))
        return results

    def compare(self, qa_pairs: list[QAPair]) -> list[CompareResult]:
        """QA 쌍으로 두 모델을 비교합니다."""
        if not qa_pairs:
            logger.warning("비교할 QA 쌍이 없습니다")
            return []

        samples = qa_pairs[: self.compare_config.max_samples]
        return run_async(self._compare_async(samples))

    def save_results(self, results: list[CompareResult], path: Path) -> None:
        """비교 결과를 JSON 파일로 저장합니다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in results]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("비교 결과 저장: %s (%d건)", path, len(results))

    def print_summary(self, results: list[CompareResult]) -> None:
        """비교 결과 요약을 출력합니다."""
        from rich.console import Console

        con = Console()

        if not results:
            con.print("[yellow]비교 결과가 없습니다[/yellow]")
            return

        # 메트릭 쌍 추출 (base_bleu -> bleu, finetuned_bleu -> bleu)
        metric_names: list[str] = []
        for r in results:
            for k in r.scores:
                if k.startswith("base_"):
                    name = k[5:]  # "base_" 제거
                    if name not in metric_names:
                        metric_names.append(name)

        table = Table(title="모델 비교 결과 요약")
        table.add_column("메트릭", style="cyan")
        table.add_column("Base 모델 평균", justify="right")
        table.add_column("Fine-tuned 모델 평균", justify="right")
        table.add_column("개선율", justify="right", style="bold")

        for name in metric_names:
            base_key = f"base_{name}"
            ft_key = f"finetuned_{name}"

            base_vals = [r.scores[base_key] for r in results if base_key in r.scores]
            ft_vals = [r.scores[ft_key] for r in results if ft_key in r.scores]

            if base_vals and ft_vals:
                base_avg = sum(base_vals) / len(base_vals)
                ft_avg = sum(ft_vals) / len(ft_vals)

                if base_avg > 0:
                    improvement = (ft_avg - base_avg) / base_avg * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"

                table.add_row(
                    name,
                    f"{base_avg:.4f}",
                    f"{ft_avg:.4f}",
                    improvement_str,
                )

        con.print(table)
        con.print(f"\n총 {len(results)}건 비교 완료")
