"""설정 시스템(config.py)의 단위 테스트입니다."""

import warnings
from pathlib import Path

import pytest

from slm_factory.config import (
    ChunkingConfig,
    CorpusExportConfig,
    EarlyStoppingConfig,
    EvalConfig,
    ExportConfig,
    GroundednessConfig,
    HwpxOptions,
    LoraConfig,
    OllamaExportConfig,
    ParsingConfig,
    PathsConfig,
    PdfOptions,
    ProjectConfig,
    QuantizationConfig,
    QuestionsConfig,
    RefinementConfig,
    ScoringConfig,
    SLMConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
    ValidationConfig,
    _EN_DEFAULT_OLLAMA_SYSTEM_PROMPT,
    _EN_DEFAULT_QA_SYSTEM_PROMPT,
    _KO_DEFAULT_OLLAMA_SYSTEM_PROMPT,
    _KO_DEFAULT_QA_SYSTEM_PROMPT,
    create_default_config,
    load_config,
)


# ---------------------------------------------------------------------------
# 개별 Config 클래스 기본값 검증
# ---------------------------------------------------------------------------


class TestProjectConfig:
    """ProjectConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """name, version, language의 기본값을 검증합니다."""
        cfg = ProjectConfig()
        assert cfg.name == "my-project"
        assert cfg.version == "1.0.0"
        assert cfg.language == "en"


class TestPathsConfig:
    """PathsConfig 기본값 및 ensure_dirs 테스트입니다."""

    def test_기본값(self):
        """documents와 output의 기본 경로를 검증합니다."""
        cfg = PathsConfig()
        assert cfg.documents == Path("./documents")
        assert cfg.output == Path("./output")

    def test_ensure_dirs_디렉토리_생성(self, tmp_path):
        """ensure_dirs 호출 시 디렉토리가 실제로 생성되는지 확인합니다."""
        cfg = PathsConfig(
            documents=tmp_path / "docs",
            output=tmp_path / "out",
        )
        cfg.ensure_dirs()
        assert (tmp_path / "docs").is_dir()
        assert (tmp_path / "out").is_dir()


class TestPdfOptions:
    """PdfOptions 기본값 테스트입니다."""

    def test_기본값(self):
        """extract_tables의 기본값이 True인지 확인합니다."""
        cfg = PdfOptions()
        assert cfg.extract_tables is True


class TestHwpxOptions:
    """HwpxOptions 기본값 테스트입니다."""

    def test_기본값(self):
        """apply_spacing의 기본값이 True인지 확인합니다."""
        cfg = HwpxOptions()
        assert cfg.apply_spacing is True


class TestParsingConfig:
    """ParsingConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """formats, pdf, hwpx의 기본값을 검증합니다."""
        cfg = ParsingConfig()
        assert cfg.formats == ["pdf", "txt", "html", "md", "hwpx", "hwp", "docx"]
        assert isinstance(cfg.pdf, PdfOptions)
        assert isinstance(cfg.hwpx, HwpxOptions)


class TestTeacherConfig:
    """TeacherConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """교사 설정의 기본값을 검증합니다."""
        cfg = TeacherConfig()
        assert cfg.backend == "ollama"
        assert cfg.model == "qwen3.5:9b"
        assert cfg.api_base == "http://localhost:11434"
        assert cfg.api_key is None
        assert cfg.temperature == 0.3
        assert cfg.timeout == 300
        assert cfg.max_context_chars == 12000
        assert cfg.max_concurrency == 2


class TestQuestionsConfig:
    """QuestionsConfig 기본값 및 메서드 테스트입니다."""

    def test_기본값(self):
        """categories, file, output_format의 기본값을 검증합니다."""
        cfg = QuestionsConfig()
        assert cfg.categories == {}
        assert cfg.file is None
        assert cfg.output_format == "alpaca"
        assert "domain-expert assistant" in cfg.system_prompt

    def test_get_all_questions_카테고리에서_추출(self):
        """categories 딕셔너리에서 모든 질문을 평탄화하여 반환하는지 확인합니다."""
        cfg = QuestionsConfig(
            categories={
                "factual": ["질문1", "질문2"],
                "reasoning": ["질문3"],
            }
        )
        questions = cfg.get_all_questions()
        assert len(questions) == 3
        assert "질문1" in questions
        assert "질문3" in questions

    def test_get_all_questions_파일에서_로드(self, tmp_path):
        """file 경로가 지정되면 파일에서 질문을 로드하는지 확인합니다."""
        qfile = tmp_path / "questions.txt"
        qfile.write_text("파일 질문1\n파일 질문2\n\n파일 질문3\n", encoding="utf-8")
        cfg = QuestionsConfig(file=qfile)
        questions = cfg.get_all_questions()
        assert questions == ["파일 질문1", "파일 질문2", "파일 질문3"]

    def test_get_all_questions_빈_카테고리(self):
        """카테고리가 비어있으면 빈 리스트를 반환하는지 확인합니다."""
        cfg = QuestionsConfig()
        assert cfg.get_all_questions() == []

    def test_get_questions_with_categories_카테고리_유지(self):
        """카테고리명과 질문이 튜플로 반환되는지 확인합니다."""
        cfg = QuestionsConfig(
            categories={
                "overview": ["질문1", "질문2"],
                "technical": ["질문3"],
            }
        )
        result = cfg.get_questions_with_categories()
        assert len(result) == 3
        assert ("overview", "질문1") in result
        assert ("overview", "질문2") in result
        assert ("technical", "질문3") in result

    def test_get_questions_with_categories_파일에서_로드(self, tmp_path):
        """파일에서 로드한 질문은 카테고리가 빈 문자열인지 확인합니다."""
        qfile = tmp_path / "questions.txt"
        qfile.write_text("파일 질문1\n파일 질문2\n", encoding="utf-8")
        cfg = QuestionsConfig(file=qfile)
        result = cfg.get_questions_with_categories()
        assert result == [("", "파일 질문1"), ("", "파일 질문2")]

    def test_get_questions_with_categories_빈_카테고리(self):
        """카테고리가 비어있으면 빈 리스트를 반환하는지 확인합니다."""
        cfg = QuestionsConfig()
        assert cfg.get_questions_with_categories() == []


class TestValidationConfig:
    """ValidationConfig 기본값 테스트입니다."""

    def test_기본값(self, sample_validation_config):
        """검증 설정의 기본값을 검증합니다."""
        cfg = sample_validation_config
        assert cfg.enabled is True
        assert cfg.min_answer_length == 20
        assert cfg.max_answer_length == 2000
        assert cfg.remove_empty is True
        assert cfg.deduplicate is True
        assert len(cfg.reject_patterns) == 6
        assert isinstance(cfg.groundedness, GroundednessConfig)


class TestGroundednessConfig:
    """GroundednessConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """근거 검증 설정의 기본값을 검증합니다."""
        cfg = GroundednessConfig()
        assert cfg.enabled is True
        assert cfg.model == "all-MiniLM-L6-v2"
        assert cfg.threshold == 0.3


class TestStudentConfig:
    """StudentConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """학생 모델 설정의 기본값을 검증합니다."""
        cfg = StudentConfig()
        assert cfg.model == "google/gemma-3-1b-it"
        assert cfg.max_seq_length == 4096


class TestLoraConfig:
    """LoraConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """LoRA 하이퍼파라미터 기본값을 검증합니다."""
        cfg = LoraConfig()
        assert cfg.r == 32
        assert cfg.alpha == 32
        assert cfg.dropout == 0.1
        assert cfg.target_modules == "auto"
        assert cfg.use_rslora is False


class TestTrainingConfig:
    """TrainingConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """훈련 설정의 기본값을 검증합니다."""
        cfg = TrainingConfig()
        assert cfg.batch_size == 1
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.learning_rate == "auto"
        assert cfg.lr_scheduler == "cosine"
        assert cfg.warmup_ratio == 0.1
        assert cfg.num_epochs == "auto"
        assert cfg.optimizer == "adamw_torch_fused"
        assert cfg.bf16 is True
        assert cfg.train_split == 0.9
        assert cfg.save_strategy == "epoch"
        assert cfg.weight_decay == 0.05
        assert cfg.label_smoothing_factor == 0.1
        assert cfg.neftune_noise_alpha == 5.0
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.early_stopping, EarlyStoppingConfig)
        assert isinstance(cfg.quantization, QuantizationConfig)


class TestExportConfig:
    """ExportConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """내보내기 설정의 기본값을 검증합니다."""
        cfg = ExportConfig()
        assert cfg.merge_lora is True
        assert cfg.output_format == "safetensors"
        assert isinstance(cfg.ollama, OllamaExportConfig)


class TestOllamaExportConfig:
    """OllamaExportConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """Ollama 내보내기 설정의 기본값을 검증합니다."""
        cfg = OllamaExportConfig()
        assert cfg.enabled is True
        assert cfg.model_name == "my-project-model"
        assert "helpful" in cfg.system_prompt


class TestEarlyStoppingConfig:
    """EarlyStoppingConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """조기 종료 설정의 기본값을 검증합니다."""
        cfg = EarlyStoppingConfig()
        assert cfg.enabled is True
        assert cfg.patience == 3
        assert cfg.threshold == 0.01


class TestQuantizationConfig:
    """QuantizationConfig 기본값 테스트입니다."""

    def test_기본값(self):
        """양자화 설정의 기본값을 검증합니다."""
        cfg = QuantizationConfig()
        assert cfg.enabled is True
        assert cfg.bits == 4


# ---------------------------------------------------------------------------
# SLMConfig (루트 설정)
# ---------------------------------------------------------------------------


class TestSLMConfig:
    """SLMConfig 루트 설정의 테스트입니다."""

    def test_기본값(self, default_config):
        """SLMConfig의 모든 하위 설정이 기본값으로 초기화되는지 확인합니다."""
        cfg = default_config
        assert isinstance(cfg.project, ProjectConfig)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.parsing, ParsingConfig)
        assert isinstance(cfg.teacher, TeacherConfig)
        assert isinstance(cfg.questions, QuestionsConfig)
        assert isinstance(cfg.validation, ValidationConfig)
        assert isinstance(cfg.student, StudentConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.export, ExportConfig)

    def test_strip_none_sections_None_섹션_제거(self):
        """_strip_none_sections가 None 값인 최상위 키를 제거하는지 확인합니다."""
        cfg = SLMConfig(**{"project": None, "teacher": None})
        # None 섹션이 제거되고 기본값이 적용되어야 합니다
        assert cfg.project.name == "my-project"
        assert cfg.teacher.backend == "ollama"

    def test_make_config_오버라이드(self, make_config):
        """make_config fixture로 특정 필드를 오버라이드할 수 있는지 확인합니다."""
        cfg = make_config(project={"name": "custom-project", "language": "ko"})
        assert cfg.project.name == "custom-project"
        assert cfg.project.language == "ko"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """load_config 함수의 테스트입니다."""

    def test_정상_YAML_로드(self, tmp_yaml_config):
        """유효한 YAML 파일을 로드하여 SLMConfig를 반환하는지 확인합니다."""
        cfg = load_config(tmp_yaml_config)
        assert isinstance(cfg, SLMConfig)
        assert cfg.project.name == "test-project"
        assert cfg.project.language == "ko"
        assert cfg.teacher.backend == "ollama"

    def test_존재하지_않는_파일(self, tmp_path):
        """존재하지 않는 파일 경로가 FileNotFoundError를 발생시키는지 확인합니다."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestCreateDefaultConfig:
    """create_default_config 함수의 테스트입니다."""

    def test_문자열_반환(self):
        """create_default_config가 비어있지 않은 문자열을 반환하는지 확인합니다."""
        result = create_default_config()
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------


class TestChunkingConfig:
    """ChunkingConfig 청킹 설정의 테스트입니다."""

    def test_기본값(self):
        """청킹 설정의 기본값을 검증합니다."""
        cfg = ChunkingConfig()
        assert cfg.enabled is True
        assert cfg.chunk_size == "auto"
        assert cfg.overlap_chars == 500

    def test_chunk_size_최소값_검증(self):
        """chunk_size가 1000 미만이면 ValueError를 발생시키는지 확인합니다."""
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkingConfig(enabled=True, chunk_size=500)

    def test_overlap_음수_검증(self):
        """overlap_chars가 음수이면 ValueError를 발생시키는지 확인합니다."""
        with pytest.raises(ValueError, match="overlap_chars"):
            ChunkingConfig(enabled=True, overlap_chars=-1)

    def test_overlap_chunk_size_이상_검증(self):
        """overlap_chars가 chunk_size 이상이면 ValueError를 발생시키는지 확인합니다."""
        with pytest.raises(ValueError, match="overlap_chars"):
            ChunkingConfig(enabled=True, chunk_size=2000, overlap_chars=2000)

    def test_정상값_통과(self):
        """유효한 값이 정상적으로 생성되는지 확인합니다."""
        cfg = ChunkingConfig(enabled=True, chunk_size=5000, overlap_chars=200)
        assert cfg.enabled is True
        assert cfg.chunk_size == 5000
        assert cfg.overlap_chars == 200


# ---------------------------------------------------------------------------
# ScoringConfig 재생성 필드
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SLMConfig에 chunking 필드 존재 확인
# ---------------------------------------------------------------------------


class TestSLMConfigChunking:
    """SLMConfig에 chunking 필드가 포함되는지 확인합니다."""

    def test_chunking_필드_존재(self, default_config):
        """SLMConfig에 chunking 속성이 기본 ChunkingConfig로 존재하는지 확인합니다."""
        assert isinstance(default_config.chunking, ChunkingConfig)
        assert default_config.chunking.enabled is True

    def test_chunking_오버라이드(self, make_config):
        """make_config로 chunking 설정을 오버라이드할 수 있는지 확인합니다."""
        cfg = make_config(chunking={"enabled": True, "chunk_size": 5000})
        assert cfg.chunking.enabled is True
        assert cfg.chunking.chunk_size == 5000


# ---------------------------------------------------------------------------
# 언어별 기본값 자동 전환
# ---------------------------------------------------------------------------


class TestSLMConfigLanguageDefaults:
    """SLMConfig._apply_language_defaults 검증기 테스트입니다."""

    def test_한국어_QA_시스템_프롬프트_자동_전환(self, make_config):
        """language='ko'일 때 QA 시스템 프롬프트가 한국어로 자동 전환되는지 확인합니다."""
        cfg = make_config(project={"language": "ko"})
        assert cfg.questions.system_prompt == _KO_DEFAULT_QA_SYSTEM_PROMPT
        assert "문서" in cfg.questions.system_prompt

    def test_한국어_Ollama_시스템_프롬프트_자동_전환(self, make_config):
        """language='ko'일 때 Ollama 시스템 프롬프트가 한국어로 자동 전환되는지 확인합니다."""
        cfg = make_config(project={"language": "ko"})
        assert cfg.export.ollama.system_prompt == _KO_DEFAULT_OLLAMA_SYSTEM_PROMPT
        assert "어시스턴트" in cfg.export.ollama.system_prompt

    def test_영어_기본값_유지(self, make_config):
        """language='en'일 때 영어 기본값이 그대로 유지되는지 확인합니다."""
        cfg = make_config(project={"language": "en"})
        assert cfg.questions.system_prompt == _EN_DEFAULT_QA_SYSTEM_PROMPT
        assert cfg.export.ollama.system_prompt == _EN_DEFAULT_OLLAMA_SYSTEM_PROMPT

    def test_한국어_커스텀_프롬프트_보존(self, make_config):
        """language='ko'이지만 사용자 지정 프롬프트는 덮어쓰지 않는지 확인합니다."""
        custom_prompt = "내가 직접 작성한 시스템 프롬프트입니다."
        cfg = make_config(
            project={"language": "ko"},
            questions={"system_prompt": custom_prompt},
        )
        assert cfg.questions.system_prompt == custom_prompt

    def test_한국어_커스텀_Ollama_프롬프트_보존(self, make_config):
        """language='ko'이지만 사용자 지정 Ollama 프롬프트는 덮어쓰지 않는지 확인합니다."""
        custom_prompt = "커스텀 Ollama 프롬프트"
        cfg = make_config(
            project={"language": "ko"},
            export={"ollama": {"system_prompt": custom_prompt}},
        )
        assert cfg.export.ollama.system_prompt == custom_prompt

    def test_언어_미지정_영어_기본값(self):
        """language를 지정하지 않으면 영어 기본값이 적용되는지 확인합니다."""
        cfg = SLMConfig()
        assert cfg.project.language == "en"
        assert cfg.questions.system_prompt == _EN_DEFAULT_QA_SYSTEM_PROMPT

    def test_YAML_로드_한국어_자동_전환(self, tmp_yaml_config):
        """YAML 파일에서 language='ko'를 로드하면 한국어 프롬프트가 적용되는지 확인합니다."""
        cfg = load_config(tmp_yaml_config)
        assert cfg.questions.system_prompt == _KO_DEFAULT_QA_SYSTEM_PROMPT
        assert cfg.export.ollama.system_prompt == _KO_DEFAULT_OLLAMA_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# EvalConfig llm_judge 필드
# ---------------------------------------------------------------------------


class TestEvalConfigLlmJudge:
    def test_기본_메트릭에_llm_judge_포함(self):
        cfg = EvalConfig()
        assert "llm_judge" in cfg.metrics

    def test_llm_judge_model_기본_빈문자열(self):
        cfg = EvalConfig()
        assert cfg.llm_judge_model == ""

    def test_llm_judge_model_설정(self):
        cfg = EvalConfig(llm_judge_model="qwen3.5:9b")
        assert cfg.llm_judge_model == "qwen3.5:9b"

    def test_llm_judge_없이_기존_메트릭만(self):
        cfg = EvalConfig(metrics=["bleu", "rouge"])
        assert "llm_judge" not in cfg.metrics
        assert "bleu" in cfg.metrics


# ---------------------------------------------------------------------------
# TrainingConfig num_epochs auto
# ---------------------------------------------------------------------------


class TestTrainingConfigAutoEpochs:
    def test_기본값_auto(self):
        cfg = TrainingConfig()
        assert cfg.num_epochs == "auto"

    def test_정수_설정(self):
        cfg = TrainingConfig(num_epochs=10)
        assert cfg.num_epochs == 10

    def test_auto_문자열_설정(self):
        cfg = TrainingConfig(num_epochs="auto")
        assert cfg.num_epochs == "auto"


# ---------------------------------------------------------------------------
# RefinementConfig
# ---------------------------------------------------------------------------


class TestRefinementConfig:
    def test_기본값(self):
        cfg = RefinementConfig()
        assert cfg.enabled is False
        assert cfg.max_rounds == 1
        assert cfg.llm_judge_threshold == 0.6

    def test_활성화(self):
        cfg = RefinementConfig(enabled=True, max_rounds=3, llm_judge_threshold=0.5)
        assert cfg.enabled is True
        assert cfg.max_rounds == 3
        assert cfg.llm_judge_threshold == 0.5

    def test_max_rounds_최소값_검증(self):
        with pytest.raises(ValueError, match="max_rounds"):
            RefinementConfig(max_rounds=0)

    def test_threshold_범위_검증(self):
        with pytest.raises(ValueError, match="llm_judge_threshold"):
            RefinementConfig(llm_judge_threshold=1.5)

    def test_threshold_음수_검증(self):
        with pytest.raises(ValueError, match="llm_judge_threshold"):
            RefinementConfig(llm_judge_threshold=-0.1)


class TestSLMConfigRefinement:
    def test_refinement_필드_존재(self, default_config):
        assert isinstance(default_config.refinement, RefinementConfig)
        assert default_config.refinement.enabled is False

    def test_refinement_오버라이드(self, make_config):
        cfg = make_config(refinement={"enabled": True, "max_rounds": 2})
        assert cfg.refinement.enabled is True
        assert cfg.refinement.max_rounds == 2


class TestAutoragExportMigration:
    """``autorag_export`` → ``corpus_export`` 마이그레이션 validator 테스트입니다."""

    def test_구_autorag_export_키_자동_변환(self):
        """구 autorag_export 키가 corpus_export로 마이그레이션됩니다."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = SLMConfig(
                **{
                    "autorag_export": {
                        "enabled": True,
                        "output_dir": "legacy_dir",
                        "chunk_size": 256,
                        "overlap_chars": 32,
                    }
                }
            )

        assert isinstance(cfg.corpus_export, CorpusExportConfig)
        assert cfg.corpus_export.output_dir == "legacy_dir"
        assert cfg.corpus_export.chunk_size == 256
        assert cfg.corpus_export.overlap_chars == 32

        deprecations = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecations, "DeprecationWarning이 발생해야 합니다"
        assert "autorag_export" in str(deprecations[0].message)

    def test_corpus_export_있으면_autorag_export_무시(self):
        """둘 다 있으면 corpus_export가 우선되고 autorag_export는 무시됩니다."""
        cfg = SLMConfig(
            **{
                "autorag_export": {"output_dir": "legacy_dir"},
                "corpus_export": {"output_dir": "new_dir"},
            }
        )
        assert cfg.corpus_export.output_dir == "new_dir"

    def test_corpus_export_단독_사용은_경고_없음(self):
        """신규 키 단독 사용 시 DeprecationWarning이 발생하지 않습니다."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = SLMConfig(**{"corpus_export": {"output_dir": "corpus"}})

        assert cfg.corpus_export.output_dir == "corpus"
        deprecations = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "autorag_export" in str(w.message)
        ]
        assert not deprecations
