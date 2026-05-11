"""slm-factory 테스트 스위트의 공유 fixture 정의."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Heavy ML 라이브러리 mock (torch, transformers, peft, trl, datasets 등)
# ---------------------------------------------------------------------------


def _ensure_ml_mocks() -> None:
    """테스트 환경에 없는 ML 라이브러리의 mock 모듈을 sys.modules에 등록합니다."""
    ml_modules = [
        "torch",
        "torch.cuda",
        "torch.nn",
        "transformers",
        "transformers.AutoModelForCausalLM",
        "transformers.AutoTokenizer",
        "transformers.BitsAndBytesConfig",
        "transformers.TrainingArguments",
        "transformers.EarlyStoppingCallback",
        "datasets",
        "datasets.load_dataset",
        "peft",
        "peft.LoraConfig",
        "peft.TaskType",
        "peft.PeftModel",
        "peft.get_peft_model",
        "trl",
        "trl.SFTTrainer",
        "accelerate",
        "bitsandbytes",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "sentence_transformers.util",
    ]
    for mod_name in ml_modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_ml_mocks()


# ---------------------------------------------------------------------------
# 팩토리 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def make_config():
    """SLMConfig 인스턴스를 쉽게 생성하는 팩토리 fixture입니다.

    사용법::
        cfg = make_config(project={"name": "test"})
    """
    from slm_factory.config import SLMConfig

    def _factory(**overrides) -> SLMConfig:
        return SLMConfig(**overrides)

    return _factory


@pytest.fixture
def default_config(make_config):
    """기본 설정으로 생성된 SLMConfig를 반환합니다."""
    return make_config()


@pytest.fixture
def make_qa_pair():
    """QAPair 인스턴스를 쉽게 생성하는 팩토리 fixture입니다.

    사용법::
        pair = make_qa_pair(question="Q?", answer="A.")
    """
    from slm_factory.models import QAPair

    def _factory(
        question: str = "테스트 질문입니다.",
        answer: str = "이것은 충분히 긴 테스트 답변입니다. 최소 길이를 충족시키기 위한 내용을 추가합니다.",
        instruction: str = "",
        source_doc: str = "test.pdf",
        category: str = "general",
    ) -> QAPair:
        return QAPair(
            question=question,
            answer=answer,
            instruction=instruction or question,
            source_doc=source_doc,
            category=category,
        )

    return _factory


@pytest.fixture
def make_parsed_doc():
    """ParsedDocument 인스턴스를 쉽게 생성하는 팩토리 fixture입니다.

    사용법::
        doc = make_parsed_doc(title="My Doc", content="Some content")
    """
    from slm_factory.models import ParsedDocument

    def _factory(
        doc_id: str = "test.pdf",
        title: str = "테스트 문서",
        content: str = "테스트 문서의 내용입니다. 충분한 길이의 본문을 포함합니다.",
        tables: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ParsedDocument:
        return ParsedDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            tables=tables or [],
            metadata=metadata or {},
        )

    return _factory


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """임시 텍스트 파일을 생성하여 경로를 반환합니다."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "# 샘플 문서\n\n이것은 테스트용 텍스트 파일입니다.\n", encoding="utf-8"
    )
    return f


@pytest.fixture
def tmp_md_file(tmp_path: Path) -> Path:
    """임시 마크다운 파일을 생성하여 경로를 반환합니다."""
    f = tmp_path / "readme.md"
    f.write_text(
        "# 테스트 마크다운\n\n본문 내용입니다.\n\n## 섹션 2\n\n추가 내용.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def tmp_html_file(tmp_path: Path) -> Path:
    """임시 HTML 파일을 생성하여 경로를 반환합니다."""
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><title>테스트 페이지</title></head>"
        "<body><h1>제목</h1><p>본문 내용입니다.</p>"
        "<table><tr><th>이름</th><th>값</th></tr>"
        "<tr><td>A</td><td>1</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def tmp_yaml_config(tmp_path: Path) -> Path:
    """최소한의 project.yaml 파일을 생성합니다."""
    f = tmp_path / "project.yaml"
    f.write_text(
        """\
project:
  name: "test-project"
  version: "1.0.0"
  language: "ko"

paths:
  documents: "./documents"
  output: "./output"

parsing:
  formats: ["pdf", "txt"]

teacher:
  backend: "ollama"
  model: "qwen3.5:9b"
""",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_validation_config():
    """테스트용 ValidationConfig를 반환합니다."""
    from slm_factory.config import ValidationConfig

    return ValidationConfig()
