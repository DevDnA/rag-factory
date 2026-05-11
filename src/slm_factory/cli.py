"""slm-factory용 명령줄 인터페이스입니다."""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console

from . import __version__
from .utils import get_logger

logger = get_logger("cli")

if TYPE_CHECKING:
    from .models import QAPair
    from .pipeline import Pipeline


class PipelineStep(str, enum.Enum):
    """run --until에 사용되는 파이프라인 단계입니다."""

    parse = "parse"
    generate = "generate"
    validate = "validate"
    score = "score"
    augment = "augment"
    analyze = "analyze"
    convert = "convert"
    train = "train"
    export = "export"
    eval = "eval"
    corpus_export = "corpus_export"
    rag_index = "rag_index"


app = typer.Typer(
    name="slm-factory",
    rich_markup_mode="rich",
)
console = Console()

eval_app = typer.Typer(help="모델 평가 및 비교")
tool_app = typer.Typer(help="유틸리티 도구 모음")

app.add_typer(eval_app, name="eval", rich_help_panel="📊 평가")
app.add_typer(tool_app, name="tool", rich_help_panel="🔧 도구")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"slm-factory [bold]{__version__}[/bold]")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="디버그 로그를 표시합니다"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="경고와 에러만 표시합니다"),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="버전을 표시합니다",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Teacher-Student Knowledge Distillation framework for domain-specific SLMs."""
    if verbose:
        import logging

        logging.getLogger("slm_factory").setLevel(logging.DEBUG)
    elif quiet:
        import logging

        logging.getLogger("slm_factory").setLevel(logging.WARNING)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ---------------------------------------------------------------------------
# 공통 옵션 상수
# ---------------------------------------------------------------------------

_CONFIG_HELP = (
    "프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다."
)
_RESUME_HELP = "중간 저장 파일에서 재개합니다"


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _get_error_hints(error: Exception) -> list[str]:
    """에러 유형에 따라 해결 힌트를 반환합니다."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    if isinstance(error, FileNotFoundError):
        return [
            "설정 파일을 찾을 수 없습니다. `slm-factory init`으로 프로젝트를 생성하세요"
        ]

    if (
        isinstance(error, ConnectionError)
        or "connect" in error_str
        or "connect" in error_type
        or "ollama" in error_str
        or "httpx" in error_type
    ):
        return [
            "Ollama가 실행 중인지 확인하세요: `ollama serve`",
            "모델이 다운로드되었는지 확인하세요: `ollama pull <Teacher 모델명>`",
        ]

    if isinstance(error, RuntimeError) and (
        "no documents" in error_str or "no parseable" in error_str
    ):
        return ["documents 디렉토리에 문서(PDF, TXT 등)를 추가하세요"]

    if (
        "cuda" in error_str
        or "mps" in error_str
        or "out of memory" in error_str
        or "oom" in error_str
    ):
        return [
            "GPU 메모리가 부족합니다. 다음을 시도하세요:",
            "  training.batch_size를 줄이세요 (예: 2 또는 1)",
            "  lora.r 값을 줄이세요 (예: 8)",
            "  gradient_accumulation_steps를 늘리세요",
            "  Apple Silicon: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 환경변수 설정",
        ]

    if "model" in error_str and (
        "not found" in error_str or "does not exist" in error_str
    ):
        return [
            "모델을 찾을 수 없습니다:",
            "  Ollama 모델 확인: `ollama list`",
            "  모델 다운로드: `ollama pull <모델명>`",
            "  설정 파일의 teacher.model 또는 student.model을 확인하세요",
        ]

    if "permission" in error_str or "access denied" in error_str:
        return [
            "파일/디렉토리 접근 권한이 없습니다:",
            "  출력 디렉토리의 쓰기 권한을 확인하세요",
            "  `sudo` 없이 실행 중인지 확인하세요",
        ]

    if "disk" in error_str or "no space" in error_str:
        return [
            "디스크 공간이 부족합니다:",
            "  `df -h`로 디스크 사용량을 확인하세요",
            "  불필요한 체크포인트를 삭제하세요: `slm-factory clean`",
        ]

    return ["--verbose 옵션으로 상세 로그를 확인하세요"]


def _print_error(
    title: str,
    error: Exception | str,
    hints: list[str] | None = None,
    resume_cmd: str | None = None,
) -> None:
    """사용자 친화적 에러 메시지를 Rich Panel로 출력합니다."""
    from rich.panel import Panel

    msg = f"[red]✗[/red] {title}\n\n[dim]{error}[/dim]"
    if hints:
        hint_lines = "\n".join(f"  → {hint}" for hint in hints)
        msg += f"\n\n[yellow]해결 방법:[/yellow]\n{hint_lines}"
    if resume_cmd:
        msg += f"\n\n[blue]재개하려면:[/blue]\n  [cyan]{resume_cmd}[/cyan]"
    console.print(Panel(msg, title="[red]오류[/red]", border_style="red"))


def _find_config(config_path: str) -> str:
    p = Path(config_path)
    if p.is_file():
        return config_path

    if p.name == "project.yaml":
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            candidate = parent / "project.yaml"
            if candidate.is_file():
                return str(candidate)
            if parent == Path.cwd().parent.parent:
                break

        for child in sorted(Path.cwd().iterdir()):
            if child.is_dir():
                candidate = child / "project.yaml"
                if candidate.is_file():
                    return str(candidate)

    return config_path


def _load_qa_data(
    pipeline: Pipeline,
    data: str | None,
    extra_candidates: list[str] | None = None,
) -> list[QAPair]:
    """QA 데이터를 로드합니다 (명시 경로 또는 출력 디렉토리 자동 감지).

    ``--data`` 옵션이 있으면 해당 파일을, 없으면 출력 디렉토리에서
    qa_augmented → qa_scored → qa_alpaca 순으로 탐색합니다.
    """
    from .models import QAPair

    if data is not None:
        data_path = Path(data)
        if not data_path.is_file():
            _print_error(
                "QA 데이터 파일 미발견",
                f"파일을 찾을 수 없음: {data_path}",
                ["--data 옵션의 경로를 확인하세요"],
            )
            raise typer.Exit(code=1)
        return pipeline._load_pairs(data_path)

    output_dir = pipeline.output_dir
    candidate_names = list(extra_candidates or []) + [
        "qa_augmented.json",
        "qa_scored.json",
        "qa_alpaca.json",
    ]
    for name in candidate_names:
        candidate = output_dir / name
        if candidate.is_file():
            console.print(f"[yellow]자동 감지:[/yellow] {candidate}")
            return pipeline._load_pairs(candidate)

    _print_error(
        "QA 데이터 파일 미발견",
        "출력 디렉토리에 QA 데이터 파일이 없습니다",
        ["generate 명령을 먼저 실행하거나 --data 옵션으로 경로를 지정하세요"],
    )
    raise typer.Exit(code=1)


def _load_pipeline(config_path: str) -> Pipeline:
    from pydantic import ValidationError

    from .config import load_config
    from .pipeline import Pipeline
    from .utils import setup_logging

    config_path = _find_config(config_path)
    setup_logging()

    try:
        config = load_config(config_path)
    except ValidationError as e:
        from rich.table import Table

        table = Table(title="설정 검증 오류", show_lines=True)
        table.add_column("위치", style="cyan")
        table.add_column("오류", style="red")
        table.add_column("입력값", style="yellow")
        for err in e.errors():
            loc = " → ".join(str(loc_part) for loc_part in err["loc"])
            table.add_row(loc, err["msg"], str(err.get("input", "")))
        console.print(table)
        console.print("\n[dim]ℹ 설정 파일을 확인하세요[/dim]")
        raise typer.Exit(code=1)

    return Pipeline(config)


def _try_load_parsed_docs(pipeline: Pipeline) -> list | None:
    """parsed_documents.json이 존재하면 ParsedDocument 리스트를 반환합니다.

    Resume 시 groundedness 검증을 위해 파싱된 문서를 재로드합니다.
    파일이 없으면 None을 반환하며, step_validate는 docs=None을 정상 처리합니다.
    """
    from .models import ParsedDocument

    parsed_path = pipeline.output_dir / "parsed_documents.json"
    if not parsed_path.is_file():
        return None
    try:
        raw = json.loads(parsed_path.read_text(encoding="utf-8"))
        return [ParsedDocument(**item) for item in raw]
    except (json.JSONDecodeError, TypeError, KeyError):
        import logging

        logging.getLogger("slm_factory.cli").warning(
            "파싱된 문서 파일 로드 실패: %s", parsed_path
        )
        return None


def _find_resume_point(pipeline: Pipeline) -> tuple[str, list]:
    """중간 저장 파일에서 가장 최근의 재개 지점을 탐색합니다."""
    from .models import ParsedDocument, QAPair

    output_dir = pipeline.output_dir

    augmented = output_dir / "qa_augmented.json"
    if augmented.is_file():
        pairs = pipeline._load_pairs(augmented)
        console.print(
            f"[yellow]재개 지점:[/yellow] qa_augmented.json ({len(pairs)}개 쌍)"
            " → analyze 단계부터"
        )
        return "analyze", pairs

    scored = output_dir / "qa_scored.json"
    if scored.is_file():
        pairs = pipeline._load_pairs(scored)
        console.print(
            f"[yellow]재개 지점:[/yellow] qa_scored.json ({len(pairs)}개 쌍)"
            " → augment 단계부터"
        )
        return "augment", pairs

    alpaca = output_dir / "qa_alpaca.json"
    if alpaca.is_file():
        raw = json.loads(alpaca.read_text(encoding="utf-8"))
        if raw and "question" in raw[0]:
            pairs = [QAPair(**item) for item in raw]
        elif raw and "instruction" in raw[0]:
            pairs = [
                QAPair(
                    question=item.get("instruction", ""),
                    answer=item.get("output", ""),
                    instruction=item.get("instruction", ""),
                    source_doc=item.get("source_doc", ""),
                    category=item.get("category", ""),
                )
                for item in raw
            ]
        else:
            pairs = []
        if pairs:
            console.print(
                f"[yellow]재개 지점:[/yellow] qa_alpaca.json ({len(pairs)}개 쌍)"
                " → validate 단계부터"
            )
            return "validate", pairs

    parsed = output_dir / "parsed_documents.json"
    if parsed.is_file():
        raw = json.loads(parsed.read_text(encoding="utf-8"))
        docs = [ParsedDocument(**item) for item in raw]
        if docs:
            console.print(
                f"[yellow]재개 지점:[/yellow] parsed_documents.json ({len(docs)}개 문서)"
                " → generate 단계부터"
            )
            return "generate", docs

    console.print("[yellow]재개 지점 없음 — 처음부터 실행합니다[/yellow]")
    return "start", []


# ---------------------------------------------------------------------------
# 명령어
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="🚀 시작하기")
def init(
    name: str = typer.Argument(help="프로젝트 이름"),
    path: str = typer.Option(
        ".", "--path", help="프로젝트를 생성할 상위 디렉토리입니다"
    ),
) -> None:
    """새로운 slm-factory 프로젝트를 초기화합니다."""
    from .config import create_default_config

    from rich.prompt import Confirm

    project_dir = Path(path) / name
    documents_dir = project_dir / "documents"
    output_dir = project_dir / "output"
    config_path = project_dir / "project.yaml"

    if config_path.is_file():
        console.print(
            f"\n[yellow]⚠[/yellow] 설정 파일이 이미 존재합니다: [cyan]{config_path}[/cyan]"
        )
        if not Confirm.ask("  덮어쓰시겠습니까?", default=False):
            console.print("  [dim]취소됨[/dim]")
            raise typer.Exit(code=0)

    project_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = create_default_config()
    # Ollama rejects model names containing '/' — use basename only
    model_basename = Path(name).name
    config_content = template.replace('name: "my-project"', f'name: "{name}"')
    config_content = config_content.replace(
        'model_name: "my-project-model"', f'model_name: "{model_basename}-model"'
    )

    config_path.write_text(config_content, encoding="utf-8")

    env_path = project_dir / ".env"
    if not env_path.exists():
        env_path.write_text("HF_TOKEN=\n", encoding="utf-8")

    console.print(
        f"\n[green]✓[/green] 프로젝트 '{name}'가 생성되었습니다: [cyan]{project_dir}[/cyan]\n"
    )
    console.print("프로젝트 구조:")
    console.print(f"  {project_dir}/")
    console.print(f"  {documents_dir}/")
    console.print(f"  {output_dir}/")
    console.print(f"  {config_path}")
    console.print(f"\n[bold]다음 단계:[/bold]")
    console.print(
        f"  1. [cyan]{documents_dir}[/cyan] 디렉토리에 학습할 문서(PDF, TXT 등)를 추가하세요"
    )
    console.print(f"\n[bold]실행 (택 1):[/bold]")
    console.print(
        f"  [bold cyan]slf rag[/bold cyan]               RAG 채팅 즉시 시작 (30초)"
    )
    console.print(
        f"  [bold cyan]slf tune[/bold cyan]              파인튜닝 + RAG + 채팅 (30분)\n"
    )


_STEP_ORDER = [
    "parse",
    "generate",
    "validate",
    "score",
    "augment",
    "analyze",
    "convert",
    "train",
    "export",
    "eval",
    "corpus_export",
    "rag_index",
]

_RESUME_TO_STEP_IDX: dict[str, int] = {
    "generate": 1,
    "validate": 2,
    "augment": 4,
    "analyze": 5,
    "convert": 6,
    "train": 7,
    "export": 8,
    "eval": 9,
    "corpus_export": 10,
    "rag_index": 11,
}

_STEP_OUTPUT_FILES: dict[str, str] = {
    "parse": "parsed_documents.json",
    "generate": "qa_alpaca.json",
    "score": "qa_scored.json",
    "augment": "qa_augmented.json",
    "analyze": "data_analysis.json",
    "convert": "training_data.jsonl",
}


def _load_preceding_data(
    pipeline: Pipeline,
    from_step: str,
    output_dir: Path,
) -> tuple[list, list, Path | None] | None:
    """--from 사용 시 지정 단계 실행에 필요한 이전 출력 데이터를 로드합니다."""
    from .models import ParsedDocument

    docs: list[ParsedDocument] = []
    pairs: list[QAPair] = []
    training_data_path: Path | None = None

    if from_step in ("generate",):
        parsed = output_dir / "parsed_documents.json"
        if parsed.is_file():
            raw = json.loads(parsed.read_text(encoding="utf-8"))
            docs = [ParsedDocument(**item) for item in raw]
        else:
            console.print(
                f"[red]오류:[/red] --from {from_step}을 사용하려면 parsed_documents.json이 필요합니다"
            )
            return None

    elif from_step in ("validate", "score"):
        alpaca = output_dir / "qa_alpaca.json"
        if alpaca.is_file():
            pairs = pipeline._load_pairs(alpaca)
        else:
            console.print(
                f"[red]오류:[/red] --from {from_step}을 사용하려면 qa_alpaca.json이 필요합니다"
            )
            return None

    elif from_step == "augment":
        scored = output_dir / "qa_scored.json"
        alpaca = output_dir / "qa_alpaca.json"
        if scored.is_file():
            pairs = pipeline._load_pairs(scored)
        elif alpaca.is_file():
            pairs = pipeline._load_pairs(alpaca)
        else:
            console.print(
                f"[red]오류:[/red] --from augment를 사용하려면 qa_scored.json 또는 qa_alpaca.json이 필요합니다"
            )
            return None

    elif from_step in ("analyze", "convert"):
        augmented = output_dir / "qa_augmented.json"
        scored = output_dir / "qa_scored.json"
        if augmented.is_file():
            pairs = pipeline._load_pairs(augmented)
        elif scored.is_file():
            pairs = pipeline._load_pairs(scored)
        else:
            console.print(
                f"[red]오류:[/red] --from {from_step}을 사용하려면 qa_augmented.json 또는 qa_scored.json이 필요합니다"
            )
            return None

    elif from_step == "train":
        training_jsonl = output_dir / "training_data.jsonl"
        if training_jsonl.is_file():
            training_data_path = training_jsonl
        else:
            console.print(
                "[red]오류:[/red] --from train을 사용하려면 training_data.jsonl이 필요합니다"
            )
            return None

    elif from_step == "export":
        adapter = output_dir / "checkpoints" / "adapter"
        if not adapter.is_dir():
            console.print(
                "[red]오류:[/red] --from export를 사용하려면 checkpoints/adapter가 필요합니다"
            )
            return None

    elif from_step == "eval":
        parsed = output_dir / "parsed_documents.json"
        if parsed.is_file():
            raw = json.loads(parsed.read_text(encoding="utf-8"))
            docs = [ParsedDocument(**item) for item in raw]
        for candidate in ["qa_augmented.json", "qa_scored.json", "qa_alpaca.json"]:
            qa_file = output_dir / candidate
            if qa_file.is_file():
                pairs = pipeline._load_pairs(qa_file)
                break
        if not pairs:
            console.print(
                f"[red]오류:[/red] --from {from_step}을 사용하려면 QA 데이터 파일이 필요합니다"
            )
            return None

    elif from_step in ("corpus_export", "rag_index"):
        parsed = output_dir / "parsed_documents.json"
        if parsed.is_file():
            raw = json.loads(parsed.read_text(encoding="utf-8"))
            docs = [ParsedDocument(**item) for item in raw]
        for candidate in ["qa_augmented.json", "qa_scored.json", "qa_alpaca.json"]:
            qa_file = output_dir / candidate
            if qa_file.is_file():
                pairs = pipeline._load_pairs(qa_file)
                break

    return docs, pairs, training_data_path


def _run_until_step(
    pipeline: Pipeline,
    target: str,
    resume: bool,
    *,
    from_step: str | None = None,
) -> Path | None:
    from .models import ParsedDocument

    target_idx = _STEP_ORDER.index(target)

    docs: list[ParsedDocument] | None = None
    pairs: list[QAPair] = []
    training_data_path: Path | None = None
    adapter_path: Path | None = None
    model_dir: Path | None = None
    corpus_path: Path | None = None
    start_idx = 0

    if from_step:
        start_idx = _STEP_ORDER.index(from_step)
        output_dir = Path(pipeline.config.paths.output)
        if from_step == "export":
            adapter_candidate = output_dir / "checkpoints" / "adapter"
            if not adapter_candidate.is_dir():
                adapter_candidate = output_dir / "adapter"
            if adapter_candidate.is_dir():
                adapter_path = adapter_candidate
            else:
                console.print("[red]오류:[/red] adapter 디렉토리를 찾을 수 없습니다")
                return None
        _loaded = _load_preceding_data(pipeline, from_step, output_dir)
        if _loaded is not None:
            docs, pairs, training_data_path = _loaded
    elif resume:
        resume_step, resume_data = _find_resume_point(pipeline)
        if resume_step != "start":
            start_idx = _RESUME_TO_STEP_IDX.get(resume_step, 0)
            if resume_step == "generate":
                docs = resume_data
            else:
                pairs = resume_data

    if start_idx > target_idx:
        console.print(
            f"\n[bold green]{target} 단계는 이미 완료되었습니다[/bold green]\n"
        )
        return None

    for idx in range(start_idx, target_idx + 1):
        step = _STEP_ORDER[idx]

        if not resume and idx < target_idx and step in _STEP_OUTPUT_FILES:
            output_file = Path(pipeline.config.paths.output) / _STEP_OUTPUT_FILES[step]
            if output_file.exists():
                console.print(
                    f"  [dim]⏭[/dim] {step} 단계 건너뜀 (이전 결과: {output_file.name})"
                )
                if step == "parse":
                    raw = json.loads(output_file.read_text(encoding="utf-8"))
                    docs = [ParsedDocument(**item) for item in raw]
                elif step in ("generate", "score", "augment"):
                    pairs = pipeline._load_pairs(output_file)
                elif step == "convert":
                    training_data_path = output_file
                continue

        if step == "parse":
            docs = pipeline.step_parse()
            console.print(f"  [green]✓[/green] {len(docs)}개 문서 파싱 완료")

        elif step == "generate":
            if docs is None:
                docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            console.print(f"  [green]✓[/green] {len(pairs)}개 QA 쌍 생성 완료")

        elif step == "validate":
            before = len(pairs)
            pairs = (
                pipeline.step_validate(pairs, docs=docs)
                if docs
                else pipeline.step_validate(pairs)
            )
            console.print(
                f"  [green]✓[/green] 검증 완료: {len(pairs)}개 수락, {before - len(pairs)}개 거부"
            )

        elif step == "score":
            before = len(pairs)
            pairs = pipeline.step_score(pairs, docs=docs)
            console.print(
                f"  [green]✓[/green] 점수 평가: {len(pairs)}개 통과, {before - len(pairs)}개 제거"
            )

        elif step == "augment":
            before = len(pairs)
            pairs = pipeline.step_augment(pairs)
            console.print(
                f"  [green]✓[/green] 데이터 증강: {before}개 → {len(pairs)}개"
            )

        elif step == "analyze":
            pipeline.step_analyze(pairs)
            console.print(f"  [green]✓[/green] {len(pairs)}개 QA 쌍 분석 완료")

        elif step == "convert":
            training_data_path = pipeline.step_convert(pairs)
            console.print(
                f"  [green]✓[/green] 학습 데이터 변환 완료: {training_data_path}"
            )

        elif step == "train":
            if not training_data_path:
                raise typer.BadParameter(
                    "train 단계는 convert 단계가 선행되어야 합니다"
                )
            adapter_path = pipeline.step_train(training_data_path)
            console.print(f"  [green]✓[/green] 모델 학습 완료: {adapter_path}")

        elif step == "export":
            if not adapter_path:
                raise typer.BadParameter("export 단계는 train 단계가 선행되어야 합니다")
            model_dir = pipeline.step_export(adapter_path)
            console.print(f"  [green]✓[/green] 모델 내보내기 완료: {model_dir}")

        elif step == "eval":
            ollama_cfg = pipeline.config.export.ollama
            if ollama_cfg.enabled and ollama_cfg.model_name and pairs:
                eval_results = pipeline.step_eval(pairs, ollama_cfg.model_name)
                console.print(
                    f"  [green]✓[/green] 모델 평가 완료: {len(eval_results)}개 결과"
                )
            else:
                console.print(
                    "  [dim]⏭[/dim] eval 건너뜀 (Ollama 모델 미설정 또는 QA 쌍 없음)"
                )

        elif step == "corpus_export":
            if docs is not None and pairs:
                from dataclasses import asdict as _asdict

                doc_dicts = [_asdict(d) for d in docs]
                pair_dicts = [_asdict(p) for p in pairs]
                corpus_path, _qa_path = pipeline.step_corpus_export(
                    doc_dicts,
                    pair_dicts,
                )
                if corpus_path.name:
                    console.print(
                        f"  [green]✓[/green] 코퍼스 데이터 내보내기 완료: {corpus_path}"
                    )
                else:
                    console.print("  [dim]⏭[/dim] corpus_export 건너뜀 (비활성화)")
            else:
                output_dir = Path(pipeline.config.paths.output)
                candidate = (
                    output_dir
                    / pipeline.config.corpus_export.output_dir
                    / "corpus.parquet"
                )
                if candidate.is_file():
                    corpus_path = candidate
                    console.print(
                        f"  [dim]⏭[/dim] corpus_export 건너뜀 (이전 결과 사용: {candidate.name})"
                    )
                else:
                    console.print(
                        "  [dim]⏭[/dim] corpus_export 건너뜀 (문서 또는 QA 데이터 없음)"
                    )

        elif step == "rag_index":
            if corpus_path is None:
                output_dir = Path(pipeline.config.paths.output)
                candidate = (
                    output_dir
                    / pipeline.config.corpus_export.output_dir
                    / "corpus.parquet"
                )
                if candidate.is_file():
                    corpus_path = candidate
            if corpus_path and corpus_path.is_file():
                db_path = pipeline.step_rag_index(corpus_path)
                if db_path.name:
                    console.print(f"  [green]✓[/green] RAG 인덱싱 완료: {db_path}")
                    console.print(
                        f"  [dim]RAG 서버 실행: slm-factory tool rag-serve --config project.yaml[/dim]"
                    )
                else:
                    console.print("  [dim]⏭[/dim] rag_index 건너뜀 (RAG 의존성 미설치)")
            else:
                console.print("  [dim]⏭[/dim] rag_index 건너뜀 (corpus.parquet 없음)")

    return model_dir


def _start_rag_server(config) -> None:
    import socket

    db_path = Path(config.paths.output) / config.rag.vector_db_path
    if not db_path.is_dir():
        console.print(
            "\n[yellow]RAG 서버를 시작할 수 없습니다[/yellow] — "
            "Qdrant DB가 없습니다. rag_index 단계를 먼저 실행하세요.\n"
        )
        return

    try:
        from .rag.server import run_server
    except ImportError:
        console.print(
            "\n[yellow]RAG 서버 의존성 미설치[/yellow] — "
            "uv sync --extra rag --extra validation\n"
        )
        return

    port = config.rag.server_port

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sock.connect_ex(("localhost", port)) == 0:
            console.print(f"\n[yellow]포트 {port}이(가) 이미 사용 중입니다.[/yellow]\n")
            kill = typer.confirm(
                f"포트 {port}의 기존 프로세스를 종료하고 새로 시작할까요?",
                default=True,
            )
            if kill:
                import subprocess
                import time

                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True,
                )
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)

                for _ in range(20):
                    check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        if check.connect_ex(("localhost", port)) != 0:
                            break
                    finally:
                        check.close()
                    time.sleep(0.5)
                else:
                    console.print(
                        f"  [red]포트 {port}을(를) 해제하지 못했습니다.[/red]\n"
                    )
                    return

                console.print(f"  [green]포트 {port} 해제 완료[/green]\n")
            else:
                console.print("  서버 시작을 취소했습니다.\n")
                return
    finally:
        sock.close()

    ollama_model = config.rag.ollama_model or config.teacher.model
    config.rag.ollama_model = ollama_model
    console.print(
        f"\n[bold]RAG API 서버 시작[/bold]\n"
        f"  모델:   [cyan]{ollama_model}[/cyan]\n"
        f"  벡터DB: [cyan]{db_path}[/cyan]\n"
        f"  채팅:   [bold cyan]http://localhost:{port}/chat[/bold cyan]\n"
        f"  API:    [cyan]http://localhost:{port}/v1/query[/cyan]\n\n"
        f"[dim]종료: Ctrl+C[/dim]\n"
    )
    run_server(config)


@app.command(rich_help_panel="⚙️ 파이프라인")
def tune(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    resume: bool = typer.Option(False, "--resume", "-r", help=_RESUME_HELP),
    until: Optional[PipelineStep] = typer.Option(
        None, "--until", help="지정된 단계까지만 실행"
    ),
    from_step: Optional[PipelineStep] = typer.Option(
        None, "--from", help="지정된 단계부터 실행 (이전 단계의 출력 파일 필요)"
    ),
    chat: bool = typer.Option(
        True,
        "--chat/--no-chat",
        help="파이프라인 완료 후 RAG 웹 채팅을 자동으로 시작합니다",
    ),
) -> None:
    """파이프라인을 실행합니다. 완료 후 RAG 웹 채팅이 자동 시작됩니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        target_step = until.value if until is not None else "rag_index"
        label = (
            f"{target_step} 단계까지 실행 중..."
            if until
            else "전체 파이프라인 시작 중..."
        )
        if from_step:
            label = f"{from_step.value} 단계부터 " + label
        console.print(f"\n[bold blue]slm-factory[/bold blue] — {label}\n")

        model_dir = _run_until_step(
            pipeline,
            target_step,
            resume,
            from_step=from_step.value if from_step else None,
        )

        if until:
            console.print(f"\n[bold green]{target_step} 단계까지 완료![/bold green]\n")
        else:
            console.print(
                f"\n[bold green]파이프라인 완료![/bold green] 모델 저장 위치: [cyan]{model_dir}[/cyan]\n"
            )

        if chat:
            _start_rag_server(pipeline.config)

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("파이프라인 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


def _qdrant_collection_ready(db_path: Path, collection_name: str) -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(db_path))
        try:
            count = client.count(collection_name=collection_name).count
            return count > 0
        except (ValueError, Exception):
            return False
        finally:
            client.close()
    except ImportError:
        return False


@app.command(rich_help_panel="🚀 시작하기")
def rag(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    chat: bool = typer.Option(
        True,
        "--chat/--no-chat",
        help="RAG 인덱스 구축 후 웹 채팅을 자동으로 시작합니다",
    ),
) -> None:
    """RAG 서비스를 시작합니다. 인덱스가 없으면 자동으로 구축합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        db_path = (
            Path(pipeline.config.paths.output) / pipeline.config.rag.vector_db_path
        )
        needs_index = not db_path.is_dir()
        if not needs_index:
            needs_index = not _qdrant_collection_ready(
                db_path, pipeline.config.rag.collection_name
            )
            if needs_index:
                console.print(
                    "[yellow]기존 인덱스가 손상되었거나 비어 있습니다. "
                    "재구축합니다...[/yellow]\n"
                )
                import shutil

                shutil.rmtree(db_path, ignore_errors=True)
        if needs_index:
            console.print(
                "\n[bold blue]slm-factory[/bold blue] — RAG 인덱스 구축 중...\n"
            )
            from dataclasses import asdict

            docs = pipeline.step_parse()
            doc_dicts = [asdict(d) for d in docs]
            corpus_path, _ = pipeline.step_corpus_export(doc_dicts, [])
            pipeline.step_rag_index(corpus_path)

        qa_path = (
            Path(pipeline.config.paths.output)
            / pipeline.config.corpus_export.output_dir
            / "qa.parquet"
        )
        if qa_path.is_file():
            try:
                from .evaluator import RetrievalEvaluator

                console.print("[bold]검색 품질 평가 중...[/bold]\n")
                ret_eval = RetrievalEvaluator(pipeline.config)
                metrics = ret_eval.evaluate(qa_path)
                ret_eval.print_summary(metrics)
                console.print()
            except Exception as e:
                logger.debug("검색 평가 건너뜀: %s", e)

        if not chat:
            console.print(
                "\n[bold green]RAG 인덱스 구축 완료![/bold green]\n"
                f"  채팅 시작: [cyan]slf rag[/cyan]\n"
            )
            return

        _start_rag_server(pipeline.config)
    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("서버 시작 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(rich_help_panel="⚙️ 파이프라인")
def train(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(
        None, "--data", help="사전 생성된 training_data.jsonl 파일 경로입니다"
    ),
    resume: bool = typer.Option(False, "--resume", "-r", help=_RESUME_HELP),
) -> None:
    """훈련 단계를 실행합니다 (선택적으로 사전 생성된 데이터 사용)."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if data is not None:
            training_data_path = Path(data)
            if not training_data_path.is_file():
                _print_error(
                    "훈련 데이터 미발견",
                    f"파일을 찾을 수 없음: {training_data_path}",
                    ["--data 옵션의 경로를 확인하세요"],
                )
                raise typer.Exit(code=1)
        elif resume:
            step, loaded = _find_resume_point(pipeline)
            if step == "generate":
                docs = loaded
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            elif step == "validate":
                docs = _try_load_parsed_docs(pipeline)
                pairs = pipeline.step_validate(loaded, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            elif step == "augment":
                pairs = pipeline.step_augment(loaded)
            elif step == "analyze":
                pairs = loaded
            else:
                docs = pipeline.step_parse()
                pairs = pipeline.step_generate(docs)
                pairs = pipeline.step_validate(pairs, docs=docs)
                pairs = pipeline.step_score(pairs)
                pairs = pipeline.step_augment(pairs)
            pipeline.step_analyze(pairs)
            training_data_path = pipeline.step_convert(pairs)
        else:
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
            pairs = pipeline.step_validate(pairs, docs=docs)
            pairs = pipeline.step_score(pairs)
            pairs = pipeline.step_augment(pairs)
            pipeline.step_analyze(pairs)
            training_data_path = pipeline.step_convert(pairs)

        adapter_path = pipeline.step_train(training_data_path)

        console.print(
            f"\n[bold green]훈련 완료![/bold green] "
            f"어댑터 저장 위치: [cyan]{adapter_path}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("훈련 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(rich_help_panel="🚀 시작하기")
def check(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """프로젝트 설정과 환경을 사전 점검합니다."""
    from rich.table import Table

    table = Table(title="slm-factory 환경 점검")
    table.add_column("항목", style="cyan")
    table.add_column("상태", style="bold")
    table.add_column("상세", style="dim")

    all_ok = True

    try:
        from .config import load_config

        resolved = _find_config(config)
        cfg = load_config(resolved)
        table.add_row("설정 파일", "[green]OK[/green]", str(resolved))
    except Exception as e:
        cfg = None
        table.add_row("설정 파일", "[red]FAIL[/red]", str(e))
        all_ok = False

    if cfg is None:
        console.print()
        console.print(table)
        raise typer.Exit(code=1)

    doc_dir = Path(cfg.paths.documents)
    if doc_dir.is_dir():
        doc_files = [f for f in doc_dir.iterdir() if f.is_file()]
        if doc_files:
            table.add_row(
                "문서 디렉토리",
                "[green]OK[/green]",
                f"{len(doc_files)}개 파일 ({doc_dir})",
            )
        else:
            table.add_row(
                "문서 디렉토리",
                "[yellow]WARN[/yellow]",
                f"디렉토리는 있으나 파일 없음 ({doc_dir})",
            )
            all_ok = False
    else:
        table.add_row("문서 디렉토리", "[red]FAIL[/red]", f"디렉토리 없음: {doc_dir}")
        all_ok = False

    out_dir = Path(cfg.paths.output)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".check_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        table.add_row("출력 디렉토리", "[green]OK[/green]", f"쓰기 가능 ({out_dir})")
    except Exception as e:
        table.add_row("출력 디렉토리", "[red]FAIL[/red]", str(e))
        all_ok = False

    if cfg.teacher.backend == "ollama":
        import httpx

        api_base = cfg.teacher.api_base.rstrip("/")
        try:
            resp = httpx.get(f"{api_base}/api/version", timeout=5)
            if resp.status_code == 200:
                ver = resp.json().get("version", "unknown")
                table.add_row(
                    "Ollama 연결",
                    "[green]OK[/green]",
                    f"v{ver} ({api_base})",
                )
            else:
                table.add_row(
                    "Ollama 연결",
                    "[red]FAIL[/red]",
                    f"HTTP {resp.status_code}",
                )
                all_ok = False
        except Exception as e:
            table.add_row("Ollama 연결", "[red]FAIL[/red]", f"연결 불가: {e}")
            all_ok = False

        try:
            resp = httpx.get(f"{api_base}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                target = cfg.teacher.model
                if any(target in m for m in models):
                    table.add_row("모델 사용 가능", "[green]OK[/green]", target)
                else:
                    available = ", ".join(models[:5]) if models else "없음"
                    table.add_row(
                        "모델 사용 가능",
                        "[yellow]WARN[/yellow]",
                        f"'{target}' 미발견 (사용 가능: {available})",
                    )
                    all_ok = False
            else:
                table.add_row(
                    "모델 사용 가능",
                    "[red]FAIL[/red]",
                    f"모델 목록 조회 실패 (HTTP {resp.status_code})",
                )
                all_ok = False
        except Exception:
            table.add_row(
                "모델 사용 가능",
                "[yellow]WARN[/yellow]",
                "모델 목록 조회 불가 (Ollama 연결 필요)",
            )
    else:
        table.add_row(
            "LLM 백엔드",
            "[green]OK[/green]",
            f"{cfg.teacher.backend} ({cfg.teacher.api_base})",
        )

    # ── 학생 모델 접근 점검 ─────────────────────────────────────
    try:
        from huggingface_hub import model_info

        model_id = cfg.student.model
        model_info(model_id)
        table.add_row("학생 모델", "[green]OK[/green]", model_id)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "gated" in msg:
            table.add_row(
                "학생 모델",
                "[yellow]WARN[/yellow]",
                f"접근 제한 모델: {cfg.student.model}. `huggingface-cli login` 필요",
            )
        else:
            table.add_row(
                "학생 모델",
                "[yellow]WARN[/yellow]",
                f"{cfg.student.model}: {e}",
            )
        all_ok = False

    # ── 컴퓨팅 디바이스 감지 ─────────────────────────────────────
    try:
        from .device import detect_device

        device = detect_device()

        type_label = {
            "cuda": "[green]NVIDIA GPU (CUDA)[/green]",
            "mps": "[green]Apple Silicon GPU (MPS)[/green]",
            "cpu": "[yellow]CPU[/yellow]",
        }
        table.add_row(
            "컴퓨팅 디바이스",
            type_label.get(device.type, device.type),
            device.name,
        )

        precision = {
            "bfloat16": "bfloat16 (bf16)",
            "float16": "float16 (fp16)",
            "float32": "float32",
        }
        table.add_row(
            "학습 정밀도",
            "[green]OK[/green]" if device.is_gpu else "[yellow]WARN[/yellow]",
            precision.get(device.dtype_name, device.dtype_name),
        )

        if device.type == "cuda":
            bnb_status = (
                "[green]OK[/green]"
                if device.quantization_available
                else "[yellow]WARN[/yellow]"
            )
            bnb_detail = (
                "사용 가능"
                if device.quantization_available
                else "미설치 (uv sync --extra cuda)"
            )
            table.add_row("4bit 양자화", bnb_status, bnb_detail)
        elif device.type == "mps":
            table.add_row(
                "4bit 양자화",
                "[dim]N/A[/dim]",
                "Apple Silicon — Unified Memory로 대체",
            )

        if not device.is_gpu:
            all_ok = False
    except Exception:
        table.add_row(
            "컴퓨팅 디바이스",
            "[yellow]WARN[/yellow]",
            "PyTorch 미설치 (학습 시 필요)",
        )

    console.print()
    console.print(table)
    if all_ok:
        console.print("\n[bold green]모든 점검 통과![/bold green]")
    else:
        console.print("\n[bold yellow]일부 항목에 주의가 필요합니다.[/bold yellow]")
        console.print("\n[dim]일반적인 해결 방법:[/dim]")
        console.print("  문서 추가  → documents/ 디렉토리에 PDF, TXT 등을 넣으세요")
        console.print("  Ollama 실행 → 별도 터미널에서 [cyan]ollama serve[/cyan]")
        console.print(
            f"  모델 다운로드 → [cyan]ollama pull {cfg.teacher.model}[/cyan]\n"
        )
        raise typer.Exit(code=1)


@app.command(rich_help_panel="ℹ️ 정보")
def status(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """파이프라인 진행 상태를 확인합니다."""
    from rich.table import Table

    from .config import load_config

    try:
        cfg = load_config(_find_config(config))
    except Exception as e:
        _print_error("설정 로드 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)

    output_dir = Path(cfg.paths.output)

    stages: list[tuple[str, str, str]] = [
        ("parse", "parsed_documents.json", "문서"),
        ("generate", "qa_alpaca.json", "쌍"),
        ("score", "qa_scored.json", "쌍"),
        ("augment", "qa_augmented.json", "쌍"),
        ("analyze", "data_analysis.json", "항목"),
        ("convert", "training_data.jsonl", "줄"),
        ("train", "checkpoints/adapter/", ""),
        ("export", "merged_model/", ""),
    ]

    table = Table(title="파이프라인 진행 상태")
    table.add_column("단계", style="cyan")
    table.add_column("파일", style="dim")
    table.add_column("상태", style="bold")
    table.add_column("건수")

    for stage_name, filename, unit in stages:
        filepath = output_dir / filename.rstrip("/")
        if filename.endswith("/"):
            if filepath.is_dir():
                table.add_row(stage_name, filename, "[green]존재[/green]", "디렉토리")
            else:
                table.add_row(stage_name, filename, "[red]없음[/red]", "-")
        elif filename.endswith(".jsonl"):
            if filepath.is_file():
                with filepath.open(encoding="utf-8") as fh:
                    line_count = sum(1 for _ in fh)
                table.add_row(
                    stage_name,
                    filename,
                    "[green]존재[/green]",
                    f"{line_count}개 {unit}",
                )
            else:
                table.add_row(stage_name, filename, "[red]없음[/red]", "-")
        elif filepath.is_file():
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                count = len(data) if isinstance(data, list) else 1
                table.add_row(
                    stage_name,
                    filename,
                    "[green]존재[/green]",
                    f"{count}개 {unit}",
                )
            except Exception:
                logger.debug("Failed to parse %s", filepath, exc_info=True)
                table.add_row(stage_name, filename, "[green]존재[/green]", "?")
        else:
            table.add_row(stage_name, filename, "[red]없음[/red]", "-")

    console.print()
    console.print(table)

    if (output_dir / "qa_augmented.json").is_file():
        resume_stage = "analyze"
    elif (output_dir / "qa_scored.json").is_file():
        resume_stage = "augment"
    elif (output_dir / "qa_alpaca.json").is_file():
        resume_stage = "validate"
    elif (output_dir / "parsed_documents.json").is_file():
        resume_stage = "generate"
    else:
        resume_stage = "parse"

    all_complete = all((output_dir / fn.rstrip("/")).exists() for _, fn, _ in stages)
    if all_complete:
        console.print("\n[bold green]모든 단계가 완료되었습니다[/bold green]\n")
    else:
        console.print(
            f"\n다음 [cyan]--resume[/cyan] 실행 시 "
            f"[bold]{resume_stage}[/bold]부터 재개됩니다\n"
        )


@app.command(rich_help_panel="ℹ️ 정보")
def clean(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    all_files: bool = typer.Option(False, "--all", help="모든 출력 파일을 삭제합니다"),
    yes: bool = typer.Option(False, "--yes", "-y", help="확인 없이 삭제합니다"),
) -> None:
    """중간 생성 파일을 정리합니다."""
    import shutil

    from rich.table import Table

    from .config import load_config

    try:
        cfg = load_config(_find_config(config))
    except Exception as e:
        _print_error("설정 로드 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)

    output_dir = Path(cfg.paths.output)

    if all_files:
        targets = list(output_dir.iterdir()) if output_dir.is_dir() else []
    else:
        intermediate_names = [
            "qa_scored.json",
            "qa_augmented.json",
            "data_analysis.json",
        ]
        targets = [
            output_dir / name
            for name in intermediate_names
            if (output_dir / name).exists()
        ]

    if not targets:
        console.print("\n삭제할 파일이 없습니다.\n")
        return

    console.print("\n[bold]삭제 대상:[/bold]")
    for t in targets:
        console.print(f"  {t}")

    if not yes:
        typer.confirm("\n삭제하시겠습니까?", abort=True)

    deleted: list[Path] = []
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            deleted.append(t)
        elif t.is_file():
            t.unlink()
            deleted.append(t)

    table = Table(title="삭제 결과")
    table.add_column("파일", style="cyan")
    table.add_column("상태", style="bold")
    for d in deleted:
        table.add_row(str(d), "[green]삭제됨[/green]")

    console.print()
    console.print(table)
    console.print(f"\n[bold green]{len(deleted)}개 항목 삭제 완료[/bold green]\n")


@tool_app.command(name="compare-data")
def compare_data(
    baseline: str = typer.Option(
        ..., "--baseline", "-b", help="기준 QA 데이터 파일 경로"
    ),
    target: str = typer.Option(
        ..., "--target", "-t", help="비교 대상 QA 데이터 파일 경로"
    ),
) -> None:
    """두 QA 데이터 세트의 품질을 비교합니다."""
    from rich.panel import Panel
    from rich.table import Table

    from .analyzer import DataAnalyzer
    from .models import QAPair

    baseline_path = Path(baseline)
    target_path = Path(target)

    if not baseline_path.is_file():
        _print_error("파일 없음", f"기준 파일을 찾을 수 없습니다: {baseline_path}")
        raise typer.Exit(code=1)
    if not target_path.is_file():
        _print_error("파일 없음", f"비교 대상 파일을 찾을 수 없습니다: {target_path}")
        raise typer.Exit(code=1)

    try:
        import json as _json
        from dataclasses import fields as _fields

        _known = {f.name for f in _fields(QAPair)}
        _alpaca_map = {"instruction": "question", "output": "answer"}

        def _normalize(item: dict) -> dict:
            mapped = {}
            for k, v in item.items():
                mapped[_alpaca_map.get(k, k)] = v
            return mapped

        baseline_data = _json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_pairs = [
            QAPair(**{k: v for k, v in _normalize(item).items() if k in _known})
            for item in baseline_data
        ]

        target_data = _json.loads(target_path.read_text(encoding="utf-8"))
        target_pairs = [
            QAPair(**{k: v for k, v in _normalize(item).items() if k in _known})
            for item in target_data
        ]
    except Exception as e:
        _print_error("파일 로드 실패", e)
        raise typer.Exit(code=1)

    analyzer = DataAnalyzer()
    baseline_report = analyzer.analyze(baseline_pairs)
    target_report = analyzer.analyze(target_pairs)

    console.print()
    console.print(Panel("[bold cyan]QA 데이터 품질 비교[/bold cyan]", expand=False))

    overview = Table(title="기본 통계 비교")
    overview.add_column("항목", style="bold")
    overview.add_column("기준 (Baseline)", justify="right")
    overview.add_column("대상 (Target)", justify="right")
    overview.add_column("변화", justify="right", style="bold")

    def _delta_str(base_val: float, target_val: float) -> str:
        diff = target_val - base_val
        if diff == 0:
            return "[dim]0[/dim]"
        sign = "+" if diff > 0 else ""
        color = "green" if diff > 0 else "red"
        return f"[{color}]{sign}{diff:.1f}[/{color}]"

    overview.add_row(
        "전체 QA 쌍",
        str(baseline_report.total_pairs),
        str(target_report.total_pairs),
        _delta_str(baseline_report.total_pairs, target_report.total_pairs),
    )
    overview.add_row(
        "원본 QA 쌍",
        str(baseline_report.original_pairs),
        str(target_report.original_pairs),
        _delta_str(baseline_report.original_pairs, target_report.original_pairs),
    )
    overview.add_row(
        "증강 QA 쌍",
        str(baseline_report.augmented_pairs),
        str(target_report.augmented_pairs),
        _delta_str(baseline_report.augmented_pairs, target_report.augmented_pairs),
    )
    overview.add_row(
        "문서 소스 수",
        str(len(baseline_report.source_doc_distribution)),
        str(len(target_report.source_doc_distribution)),
        _delta_str(
            len(baseline_report.source_doc_distribution),
            len(target_report.source_doc_distribution),
        ),
    )
    console.print(overview)

    length_table = Table(title="길이 통계 비교 (문자 수)")
    length_table.add_column("항목", style="bold")
    length_table.add_column("기준 평균", justify="right")
    length_table.add_column("대상 평균", justify="right")
    length_table.add_column("변화", justify="right", style="bold")

    for label, b_stats, t_stats in [
        (
            "답변 길이",
            baseline_report.answer_length_stats,
            target_report.answer_length_stats,
        ),
        (
            "질문 길이",
            baseline_report.question_length_stats,
            target_report.question_length_stats,
        ),
    ]:
        b_mean = b_stats.get("mean", 0.0)
        t_mean = t_stats.get("mean", 0.0)
        length_table.add_row(
            label, f"{b_mean:.1f}", f"{t_mean:.1f}", _delta_str(b_mean, t_mean)
        )

    console.print(length_table)

    all_categories = sorted(
        set(baseline_report.category_distribution)
        | set(target_report.category_distribution)
    )
    if all_categories:
        cat_table = Table(title="카테고리 분포 비교")
        cat_table.add_column("카테고리", style="bold")
        cat_table.add_column("기준", justify="right")
        cat_table.add_column("대상", justify="right")
        cat_table.add_column("변화", justify="right", style="bold")
        for cat in all_categories:
            b_count = baseline_report.category_distribution.get(cat, 0)
            t_count = target_report.category_distribution.get(cat, 0)
            cat_table.add_row(
                cat, str(b_count), str(t_count), _delta_str(b_count, t_count)
            )
        console.print(cat_table)

    all_warnings = []
    for w in target_report.warnings:
        if w not in baseline_report.warnings:
            all_warnings.append(f"[yellow]⚠ (대상) {w}[/yellow]")
    for w in baseline_report.warnings:
        if w not in target_report.warnings:
            all_warnings.append(f"[green]✓ (해결됨) {w}[/green]")
    if all_warnings:
        console.print()
        for w in all_warnings:
            console.print(f"  {w}")

    console.print(
        f"\n  기준: [cyan]{baseline_path.name}[/cyan] ({baseline_report.total_pairs}개)"
        f"\n  대상: [cyan]{target_path.name}[/cyan] ({target_report.total_pairs}개)\n"
    )


@tool_app.command(name="convert")
def convert(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(
        None,
        "--data",
        help="QA 데이터 파일 경로 (qa_alpaca.json 또는 qa_augmented.json)",
    ),
) -> None:
    """QA 데이터를 훈련용 JSONL 형식으로 변환합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        training_data_path = pipeline.step_convert(pairs)

        console.print(
            f"\n[bold green]변환 완료![/bold green] "
            f"훈련 데이터: [cyan]{training_data_path}[/cyan] ({len(pairs)}개 쌍)\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("변환 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(name="export", rich_help_panel="⚙️ 파이프라인")
def export_model(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", help="어댑터 디렉토리 경로"
    ),
) -> None:
    """훈련된 모델을 내보냅니다 (LoRA 병합 + Ollama Modelfile)."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if adapter is not None:
            adapter_path = Path(adapter)
            if not adapter_path.is_dir():
                _print_error(
                    "어댑터 디렉토리 미발견",
                    f"디렉토리를 찾을 수 없음: {adapter_path}",
                    ["--adapter 옵션의 경로를 확인하세요"],
                )
                raise typer.Exit(code=1)
        else:
            adapter_path = pipeline.output_dir / "checkpoints" / "adapter"
            if not adapter_path.is_dir():
                _print_error(
                    "어댑터 디렉토리 미발견",
                    f"디렉토리를 찾을 수 없음: {adapter_path}",
                    [
                        "--adapter 옵션으로 경로를 지정하거나 train 명령을 먼저 실행하세요"
                    ],
                )
                raise typer.Exit(code=1)

        model_dir = pipeline.step_export(adapter_path)

        console.print(
            f"\n[bold green]내보내기 완료![/bold green] "
            f"모델 저장 위치: [cyan]{model_dir}[/cyan]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("내보내기 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@app.command(rich_help_panel="ℹ️ 정보")
def version() -> None:
    """slm-factory 버전을 표시합니다."""
    console.print(f"slm-factory [bold]{__version__}[/bold]")


@eval_app.command(name="run")
def eval_model(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    model: str = typer.Option(..., "--model", help="평가할 Ollama 모델 이름입니다"),
    data: Optional[str] = typer.Option(
        None,
        "--data",
        help="QA 데이터 파일 경로 (미지정 시 출력 디렉토리에서 자동 감지)",
    ),
) -> None:
    """학습된 모델을 QA 데이터로 평가합니다."""
    try:
        from .evaluator import ModelEvaluator

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        evaluator = ModelEvaluator(pipeline.config)
        results = evaluator.evaluate(pairs, model)

        eval_output = pipeline.output_dir / pipeline.config.eval.output_file
        evaluator.save_results(results, eval_output)
        evaluator.print_summary(results)

        console.print(
            f"\n[bold green]평가 완료![/bold green] "
            f"결과: [cyan]{eval_output}[/cyan] ({len(results)}건)\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("평가 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="export-corpus")
def export_corpus(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    qa_file: Optional[str] = typer.Option(
        None,
        "--qa-file",
        help="QA 파일 경로 (기본값: 자동 감지 — scored > validated > alpaca 순)",
    ),
) -> None:
    """파싱된 문서와 QA 쌍을 외부 평가용 parquet 파일로 변환합니다."""
    try:
        from .exporter.corpus_export import CorpusExporter

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()
        output_dir = pipeline.config.paths.output

        docs_path = output_dir / "parsed_documents.json"
        if not docs_path.is_file():
            _print_error(
                "파싱 데이터 미발견",
                f"파일을 찾을 수 없음: {docs_path}",
                ["먼저 파이프라인을 실행하세요: slf tune"],
            )
            raise typer.Exit(code=1)

        parsed_docs = json.loads(docs_path.read_text(encoding="utf-8"))

        if qa_file is not None:
            qa_path = Path(qa_file)
        else:
            for candidate in [
                "qa_scored.json",
                "qa_validated.json",
                "qa_alpaca.json",
            ]:
                qa_path = output_dir / candidate
                if qa_path.is_file():
                    break
            else:
                _print_error(
                    "QA 데이터 미발견",
                    f"QA 파일을 찾을 수 없음: {output_dir}",
                    [
                        "먼저 파이프라인을 실행하세요: slf tune",
                        "또는 --qa-file 옵션으로 QA 파일 경로를 직접 지정하세요",
                    ],
                )
                raise typer.Exit(code=1)

        if not qa_path.is_file():
            _print_error(
                "QA 파일 미발견",
                f"파일을 찾을 수 없음: {qa_path}",
            )
            raise typer.Exit(code=1)

        qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))

        console.print(
            f"\n[bold]코퍼스 데이터 내보내기[/bold]\n"
            f"  문서: [cyan]{docs_path.name}[/cyan] ({len(parsed_docs)}건)\n"
            f"  QA:   [cyan]{qa_path.name}[/cyan] ({len(qa_pairs)}건)\n"
        )

        exporter = CorpusExporter(pipeline.config)
        corpus_path, qa_parquet_path = exporter.export(parsed_docs, qa_pairs)

        console.print(
            f"\n[bold green]코퍼스 내보내기 완료![/bold green]\n"
            f"  corpus: [cyan]{corpus_path}[/cyan]\n"
            f"  qa:     [cyan]{qa_parquet_path}[/cyan]\n\n"
            f"[dim]다음 단계: slf tool rag-index --config {config}[/dim]\n"
        )

    except FileNotFoundError as e:
        _print_error("파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("코퍼스 내보내기 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="eval-retrieval")
def eval_retrieval(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    top_k: int = typer.Option(5, "--top-k", help="검색 결과 상위 K개 기준"),
    qa_file: Optional[str] = typer.Option(
        None, "--qa-file", help="qa.parquet 경로 (미지정 시 자동 감지)"
    ),
) -> None:
    """검색 품질을 Recall@K, MRR, Hit@K 메트릭으로 평가합니다."""
    try:
        from .evaluator import RetrievalEvaluator

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if qa_file:
            qa_path = Path(qa_file)
        else:
            qa_path = (
                Path(pipeline.config.paths.output)
                / pipeline.config.corpus_export.output_dir
                / "qa.parquet"
            )

        if not qa_path.is_file():
            _print_error(
                "QA 데이터 미발견",
                f"파일을 찾을 수 없음: {qa_path}",
                [
                    "먼저 코퍼스 내보내기를 실행하세요: slf tool export-corpus",
                ],
            )
            raise typer.Exit(code=1)

        db_path = (
            Path(pipeline.config.paths.output) / pipeline.config.rag.vector_db_path
        )
        if not db_path.exists():
            _print_error(
                "벡터 DB 미발견",
                f"Qdrant 인덱스를 찾을 수 없음: {db_path}",
                ["먼저 인덱싱을 실행하세요: slf tool rag-index"],
            )
            raise typer.Exit(code=1)

        console.print(
            f"\n[bold]검색 품질 평가[/bold]\n"
            f"  QA 데이터: [cyan]{qa_path}[/cyan]\n"
            f"  벡터 DB:   [cyan]{db_path}[/cyan]\n"
            f"  top_k:     [cyan]{top_k}[/cyan]\n"
        )

        evaluator = RetrievalEvaluator(pipeline.config)
        metrics = evaluator.evaluate(qa_path, top_k=top_k)
        evaluator.print_summary(metrics)

        console.print(f"\n[bold green]검색 평가 완료![/bold green]\n")

    except FileNotFoundError as e:
        _print_error("파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("검색 평가 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="rag-index")
def rag_index(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    corpus_dir: Optional[str] = typer.Option(
        None,
        "--corpus-dir",
        help="corpus.parquet이 있는 디렉토리 경로 (기본값: output/corpus)",
    ),
) -> None:
    """corpus.parquet을 Qdrant에 임베딩하여 적재합니다."""
    try:
        from .rag.indexer import RAGIndexer

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        if corpus_dir is not None:
            corpus_path = Path(corpus_dir) / "corpus.parquet"
        else:
            corpus_path = (
                pipeline.config.paths.output
                / pipeline.config.corpus_export.output_dir
                / "corpus.parquet"
            )

        if not corpus_path.is_file():
            _print_error(
                "코퍼스 데이터 미발견",
                f"파일을 찾을 수 없음: {corpus_path}",
                [
                    "먼저 코퍼스 데이터를 내보내세요: "
                    "slm-factory tool export-corpus --config project.yaml",
                ],
            )
            raise typer.Exit(code=1)

        console.print(
            f"\n[bold]RAG 인덱싱[/bold]\n"
            f"  코퍼스: [cyan]{corpus_path}[/cyan]\n"
            f"  임베딩: [cyan]{pipeline.config.rag.embedding_model}[/cyan]\n"
            f"  벡터DB: [cyan]{pipeline.config.paths.output / pipeline.config.rag.vector_db_path}[/cyan]\n"
        )

        indexer = RAGIndexer(pipeline.config)
        db_path = indexer.index(corpus_path)

        console.print(
            f"\n[bold green]인덱싱 완료![/bold green]\n"
            f"  Qdrant: [cyan]{db_path}[/cyan]\n\n"
            f"[dim]다음 단계: slm-factory tool rag-serve --config {config}[/dim]\n"
        )

    except ImportError:
        _print_error(
            "RAG 의존성 미설치",
            "qdrant-client 또는 sentence-transformers가 설치되지 않았습니다",
            ["uv sync --extra rag --extra validation"],
        )
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("RAG 인덱싱 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="rag-serve")
def rag_serve(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="서버 바인드 호스트 (기본값: 설정 파일 값)",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="서버 포트 (기본값: 설정 파일 값)",
    ),
) -> None:
    """RAG API 서버를 시작합니다. Qdrant 검색 + Ollama SLM 생성."""
    try:
        from .rag.server import run_server

        pipeline = _load_pipeline(config)

        if host is not None:
            pipeline.config.rag.server_host = host
        if port is not None:
            pipeline.config.rag.server_port = port

        db_path = pipeline.config.paths.output / pipeline.config.rag.vector_db_path
        if not db_path.is_dir():
            _print_error(
                "벡터DB 미발견",
                f"Qdrant DB를 찾을 수 없음: {db_path}",
                [
                    "먼저 인덱싱을 실행하세요: "
                    "slm-factory tool rag-index --config project.yaml",
                ],
            )
            raise typer.Exit(code=1)

        ollama_model = (
            pipeline.config.rag.ollama_model or pipeline.config.teacher.model
        )

        console.print(
            f"\n[bold]RAG API 서버 시작[/bold]\n"
            f"  모델:   [cyan]{ollama_model}[/cyan]\n"
            f"  벡터DB: [cyan]{db_path}[/cyan]\n"
            f"  주소:   [cyan]http://{pipeline.config.rag.server_host}:{pipeline.config.rag.server_port}[/cyan]\n\n"
            f"[dim]API 테스트: curl -X POST http://localhost:{pipeline.config.rag.server_port}/v1/query "
            f'-H "Content-Type: application/json" '
            f'-d \'{{"query": "질문"}}\'[/dim]\n\n'
            f"[dim]종료: Ctrl+C[/dim]\n"
        )

        run_server(pipeline.config)

    except ImportError:
        _print_error(
            "RAG 의존성 미설치",
            "fastapi, uvicorn 또는 qdrant-client가 설치되지 않았습니다",
            ["uv sync --extra rag --extra validation"],
        )
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("RAG 서버 실행 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="evolve")
def evolve(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    force_update: bool = typer.Option(
        False,
        "--force-update",
        help="변경 감지를 무시하고 전체 문서를 재처리합니다",
    ),
    skip_gate: bool = typer.Option(
        False,
        "--skip-gate",
        help="품질 게이트를 건너뛰고 무조건 배포합니다",
    ),
) -> None:
    """증분 업데이트 → 재학습 → 품질 게이트 → 버전된 모델 배포를 단일 명령으로 실행합니다."""
    from rich.panel import Panel

    try:
        from .evolve_history import EvolveHistory
        from .incremental import IncrementalTracker

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        history = EvolveHistory(pipeline.config)
        tracker = IncrementalTracker(pipeline.config)

        # ── 1. 증분 업데이트 ──────────────────────────────────────
        console.print("\n[bold]━━━ [1/5] 증분 업데이트 ━━━[/bold]")

        if force_update:
            console.print("  [yellow]--force-update: 전체 문서 재처리[/yellow]")
            docs = pipeline.step_parse()
            pairs = pipeline.step_generate(docs)
        else:
            changed_files = tracker.get_changed_files(pipeline.config.paths.documents)
            if not changed_files:
                console.print("  변경된 문서가 없습니다")
                console.print(
                    "\n[dim]힌트: --force-update 옵션으로 전체 재처리할 수 있습니다[/dim]\n"
                )
                return

            console.print(f"  {len(changed_files)}개 변경 문서 감지")
            docs = pipeline.step_parse(files=changed_files)
            pairs = pipeline.step_generate(docs)

        existing_path = pipeline.output_dir / "qa_alpaca.json"
        if existing_path.is_file():
            existing_pairs = pipeline._load_pairs(existing_path)
        else:
            existing_pairs = []

        strategy = pipeline.config.incremental.merge_strategy
        merged = tracker.merge_qa_pairs(existing_pairs, pairs, strategy)
        pipeline._save_pairs(merged, existing_path)
        tracker.commit_hashes()  # 처리 성공 후 해시 확정
        console.print(
            f"  [green]✓[/green] 새 QA {len(pairs)}개, 전체 {len(merged)}개 (전략: {strategy})"
        )

        # ── 2. 검증 + 점수 + 증강 ────────────────────────────────
        console.print("\n[bold]━━━ [2/5] 검증 · 점수 · 증강 ━━━[/bold]")
        pairs = pipeline.step_validate(merged, docs=docs)
        console.print(f"  [green]✓[/green] 검증: {len(pairs)}개 통과")
        pairs = pipeline.step_score(pairs)

        pairs = pipeline.step_augment(pairs)
        pipeline.step_analyze(pairs)
        console.print(f"  [green]✓[/green] 최종 학습 데이터: {len(pairs)}개")

        # ── 3. 변환 + 학습 + 내보내기 ────────────────────────────
        console.print("\n[bold]━━━ [3/5] 학습 · 내보내기 ━━━[/bold]")
        training_data_path = pipeline.step_convert(pairs)
        adapter_path = pipeline.step_train(training_data_path)

        from .exporter import HFExporter

        hf_exporter = HFExporter(pipeline.config)
        model_dir = hf_exporter.export(adapter_path)
        console.print(f"  [green]✓[/green] 모델 병합: {model_dir}")

        # ── 4. 버전 생성 + 품질 게이트 ───────────────────────────
        console.print("\n[bold]━━━ [4/5] 품질 게이트 ━━━[/bold]")
        version = history.generate_version_name()
        versioned_name = history.generate_model_name(version)
        is_first = history.is_first_run()

        if is_first or skip_gate:
            reason = "첫 실행 (비교 대상 없음)" if is_first else "--skip-gate"
            console.print(f"  [yellow]품질 게이트 건너뜀:[/yellow] {reason}")

            if pipeline.config.export.ollama.enabled:
                from .exporter import OllamaExporter

                ollama_exporter = OllamaExporter(pipeline.config)
                modelfile_path = ollama_exporter.generate_modelfile(model_dir)
                success = ollama_exporter.create_model(
                    modelfile_path,
                    model_name_override=versioned_name,
                )
                if success:
                    console.print(
                        f"  [green]✓[/green] Ollama 모델 생성: {versioned_name}"
                    )
                else:
                    console.print(
                        f"  [yellow]⚠[/yellow] Ollama 모델 생성 실패 (수동: ollama create {versioned_name} -f {modelfile_path})"
                    )

            history.record_version(
                version,
                versioned_name,
                qa_count=len(pairs),
                promoted=True,
            )
        else:
            previous_name = history.get_current_model_name()
            if previous_name is None:
                console.print("  [yellow]이전 모델 정보 없음 — 무조건 배포[/yellow]")
                gate_passed = True
                gate_scores: dict[str, float] = {}
            else:
                console.print(f"  비교: {previous_name} vs {versioned_name}")

                if pipeline.config.export.ollama.enabled:
                    from .exporter import OllamaExporter

                    ollama_exporter = OllamaExporter(pipeline.config)
                    modelfile_path = ollama_exporter.generate_modelfile(model_dir)
                    ollama_exporter.create_model(
                        modelfile_path,
                        model_name_override=versioned_name,
                    )

                pipeline.config.compare.enabled = True
                pipeline.config.compare.base_model = previous_name
                pipeline.config.compare.finetuned_model = versioned_name

                try:
                    compare_results = pipeline.step_compare(pairs)
                    gate_passed, gate_scores = history.check_quality_gate(
                        compare_results
                    )
                except Exception as e:
                    console.print(f"  [yellow]⚠[/yellow] 품질 비교 실패: {e}")
                    console.print(
                        "  [yellow]모델은 생성되었으나 품질 검증 미완료[/yellow]"
                    )
                    history.record_version(
                        version,
                        versioned_name,
                        qa_count=len(pairs),
                        promoted=False,
                    )
                    console.print(
                        Panel(
                            f"[bold yellow]진화 부분 완료[/bold yellow]\n\n"
                            f"  버전: [cyan]{version}[/cyan]\n"
                            f"  모델: [cyan]{versioned_name}[/cyan]\n"
                            f"  상태: 품질 검증 보류\n\n"
                            f"Ollama 실행 후 수동 비교:\n"
                            f"  [cyan]slm-factory eval compare --base-model {previous_name} --ft {versioned_name}[/cyan]",
                            expand=False,
                        )
                    )
                    return

            if gate_passed:
                improvement = gate_scores.get("improvement_pct", 0)
                console.print(
                    f"  [green]✓[/green] 품질 게이트 통과 ({improvement:+.1f}%)"
                )
                history.record_version(
                    version,
                    versioned_name,
                    scores=gate_scores,
                    qa_count=len(pairs),
                    promoted=True,
                )
            else:
                improvement = gate_scores.get("improvement_pct", 0)
                console.print(f"  [red]✗[/red] 품질 게이트 실패 ({improvement:+.1f}%)")
                history.record_version(
                    version,
                    versioned_name,
                    scores=gate_scores,
                    qa_count=len(pairs),
                    promoted=False,
                )
                # 품질 게이트 실패 시 생성된 Ollama 모델 정리
                EvolveHistory._ollama_rm(versioned_name)
                console.print(
                    Panel(
                        f"[bold red]진화 중단 — 품질 미달[/bold red]\n\n"
                        f"  버전: [cyan]{version}[/cyan]\n"
                        f"  모델: [cyan]{versioned_name}[/cyan] (미배포)\n"
                        f"  메트릭: {pipeline.config.evolve.gate_metric}\n"
                        f"  개선율: {improvement:+.1f}% "
                        f"(최소 {pipeline.config.evolve.gate_min_improvement}% 필요)\n\n"
                        f"현재 활성 모델: [green]{previous_name}[/green]",
                        expand=False,
                    )
                )
                return

        # ── 5. 정리 + 완료 ───────────────────────────────────────
        console.print("\n[bold]━━━ [5/5] 완료 ━━━[/bold]")
        removed = history.cleanup_old_versions()
        if removed:
            console.print(f"  이전 버전 정리: {', '.join(removed)}")

        console.print(
            Panel(
                f"[bold green]진화 완료![/bold green]\n\n"
                f"  버전: [cyan]{version}[/cyan]\n"
                f"  모델: [cyan]{versioned_name}[/cyan]\n"
                f"  QA 쌍: [cyan]{len(pairs)}[/cyan]개\n"
                f"  모델 경로: [cyan]{model_dir}[/cyan]\n\n"
                f"모델 실행:\n"
                f"  [cyan]ollama run {versioned_name}[/cyan]",
                expand=False,
            )
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("진화 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="update")
def update(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
) -> None:
    """변경된 문서만 증분 처리합니다."""
    try:
        from .incremental import IncrementalTracker

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        tracker = IncrementalTracker(pipeline.config)
        changed_files = tracker.get_changed_files(pipeline.config.paths.documents)

        if not changed_files:
            console.print("\n변경된 문서가 없습니다\n")
            return

        console.print(
            f"\n[bold blue]증분 업데이트:[/bold blue] {len(changed_files)}개 변경 문서 감지\n"
        )

        docs = pipeline.step_parse(files=changed_files)
        pairs = pipeline.step_generate(docs)

        existing_path = pipeline.output_dir / "qa_alpaca.json"
        if existing_path.is_file():
            existing_pairs = pipeline._load_pairs(existing_path)
        else:
            existing_pairs = []

        strategy = pipeline.config.incremental.merge_strategy
        merged = tracker.merge_qa_pairs(existing_pairs, pairs, strategy)

        pipeline._save_pairs(merged, existing_path)
        tracker.commit_hashes()  # 처리 성공 후 해시 확정

        console.print(
            f"\n[bold green]증분 업데이트 완료![/bold green] "
            f"변경 문서: {len(changed_files)}개, "
            f"새 QA: {len(pairs)}개, "
            f"전체 QA: {len(merged)}개 "
            f"(전략: {strategy})\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("증분 업데이트 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@eval_app.command(name="compare")
def compare_models(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    base_model: str = typer.Option(
        ..., "--base-model", help="비교 기준 모델 이름 (Ollama)"
    ),
    finetuned_model: str = typer.Option(
        ..., "--ft", help="파인튜닝된 모델 이름 (Ollama)"
    ),
    data: Optional[str] = typer.Option(None, "--data", help="QA 데이터 파일 경로"),
) -> None:
    """Base 모델과 Fine-tuned 모델의 답변을 비교합니다."""
    try:
        from .comparator import ModelComparator

        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(pipeline, data)

        pipeline.config.compare.base_model = base_model
        pipeline.config.compare.finetuned_model = finetuned_model

        comparator = ModelComparator(pipeline.config)
        results = comparator.compare(pairs)

        compare_output = pipeline.output_dir / pipeline.config.compare.output_file
        comparator.save_results(results, compare_output)
        comparator.print_summary(results)

        console.print(
            f"\n[bold green]비교 완료![/bold green] "
            f"결과: [cyan]{compare_output}[/cyan] ({len(results)}건)\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("비교 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


@tool_app.command(name="review")
def review_qa(
    config: str = typer.Option("project.yaml", "--config", help=_CONFIG_HELP),
    data: Optional[str] = typer.Option(None, "--data", help="QA 데이터 파일 경로"),
) -> None:
    """QA 쌍을 수동으로 리뷰하는 TUI를 실행합니다."""
    try:
        pipeline = _load_pipeline(config)
        pipeline.config.paths.ensure_dirs()

        pairs = _load_qa_data(
            pipeline,
            data,
            extra_candidates=["qa_reviewed.json"],
        )

        output_path = pipeline.output_dir / pipeline.config.review.output_file

        from .tui.reviewer import QAReviewerApp

        reviewer_app = QAReviewerApp(pairs=pairs, output_path=output_path)
        reviewer_app.run()

        statuses = QAReviewerApp.count_statuses(pairs)
        console.print(
            f"\n[bold green]리뷰 완료![/bold green] "
            f"승인: [green]{statuses['approved']}[/green], "
            f"거부: [red]{statuses['rejected']}[/red], "
            f"대기: [yellow]{statuses['pending']}[/yellow]\n"
        )

    except FileNotFoundError as e:
        _print_error("설정 파일 오류", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)
    except Exception as e:
        _print_error("리뷰 실행 실패", e, hints=_get_error_hints(e))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def main() -> None:
    """pyproject.toml에서 호출되는 진입점입니다."""
    app()
