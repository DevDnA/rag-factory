"""Ollama 내보내기 — Modelfile을 생성하고 선택적으로 로컬 Ollama로 가져옵니다.

병합된 HuggingFace 모델에서 Ollama 호환 Modelfile을 생성하여
`ollama create`를 통한 원클릭 배포를 가능하게 합니다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("exporter.ollama_export")


class OllamaExporter:
    """Ollama Modelfile을 생성하고 선택적으로 모델을 가져옵니다.
    
    인자:
        config: 전체 SLMConfig
    """
    
    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.ollama_config = config.export.ollama
        self.model_name = config.export.ollama.model_name
        self.system_prompt = config.export.ollama.system_prompt
        self.parameters = config.export.ollama.parameters
    
    def generate_modelfile(
        self,
        model_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Ollama Modelfile을 생성합니다.
        
        인자:
            model_dir: HuggingFace 모델 디렉토리의 경로
            output_path: Modelfile의 출력 경로 (기본값: model_dir / "Modelfile")
        
        반환값:
            생성된 Modelfile의 경로
        """
        model_dir = Path(model_dir)
        if output_path is None:
            output_path = model_dir / "Modelfile"
        output_path = Path(output_path)
        
        logger.info("Modelfile 생성 중: %s", output_path)

        # 모델 아키텍처에 맞는 chat template 감지
        chat_template = self._detect_chat_template(model_dir)

        # Modelfile 내용 구성
        lines = [
            f"FROM {model_dir}",
            "",
        ]

        # chat template 추가 (없으면 Ollama가 raw passthrough하여 깨진 출력 발생)
        if chat_template:
            lines.extend([
                f'TEMPLATE """{chat_template}"""',
                "",
            ])

        lines.extend([
            'SYSTEM """',
            self.system_prompt,
            '"""',
            "",
        ])

        # 매개변수 추가
        for param_name, param_value in self.parameters.items():
            lines.append(f"PARAMETER {param_name} {param_value}")

        # stop 토큰 추가 (chat template에서 사용하는 turn 종료 토큰)
        stop_token = self._detect_stop_token(model_dir)
        if stop_token:
            lines.append(f"PARAMETER stop {stop_token}")
        
        content = "\n".join(lines)
        
        # Modelfile 작성
        output_path.write_text(content, encoding="utf-8")
        
        logger.info("✓ Modelfile 생성됨: %s", output_path)
        
        return output_path
    
    # -- 모델 아키텍처별 Ollama Go template 매핑 --
    # Ollama는 safetensors에서 chat template을 자동 감지하지 못하는 경우가 있음.
    # 이 경우 TEMPLATE {{ .Prompt }} (raw passthrough)로 설정되어 깨진 출력 발생.
    _CHAT_TEMPLATES: dict[str, str] = {
        "Qwen2ForCausalLM": (
            "{{- range $i, $_ := .Messages }}"
            "{{- if eq .Role \"system\" }}<|im_start|>system\n"
            "{{ .Content }}<|im_end|>\n"
            "{{- else if eq .Role \"user\" }}<|im_start|>user\n"
            "{{ .Content }}<|im_end|>\n"
            "{{- else if eq .Role \"assistant\" }}<|im_start|>assistant\n"
            "{{ .Content }}<|im_end|>\n"
            "{{- end }}"
            "{{- end }}<|im_start|>assistant\n"
        ),
        "Gemma3ForCausalLM": (
            "{{- range $i, $_ := .Messages }}"
            "{{- if eq .Role \"user\" }}<start_of_turn>user\n"
            "{{ if and $.System (eq $i 0) }}{{ $.System }}\n\n{{ end }}"
            "{{ .Content }}<end_of_turn>\n"
            "{{ else if eq .Role \"assistant\" }}<start_of_turn>model\n"
            "{{ .Content }}<end_of_turn>\n"
            "{{ end }}"
            "{{- end }}<start_of_turn>model\n"
        ),
        "LlamaForCausalLM": (
            "{{- range $i, $_ := .Messages }}"
            "{{- if eq .Role \"system\" }}<|start_header_id|>system<|end_header_id|>\n\n"
            "{{ .Content }}<|eot_id|>"
            "{{- else if eq .Role \"user\" }}<|start_header_id|>user<|end_header_id|>\n\n"
            "{{ .Content }}<|eot_id|>"
            "{{- else if eq .Role \"assistant\" }}<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{{ .Content }}<|eot_id|>"
            "{{- end }}"
            "{{- end }}<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    }

    _STOP_TOKENS: dict[str, str] = {
        "Qwen2ForCausalLM": "<|im_end|>",
        "Gemma3ForCausalLM": "<end_of_turn>",
        "LlamaForCausalLM": "<|eot_id|>",
    }

    def _detect_chat_template(self, model_dir: Path) -> str | None:
        """모델 디렉토리의 config.json에서 아키텍처를 읽어 chat template을 반환합니다."""
        import json

        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.warning("config.json 없음 — chat template 감지 건너뜀")
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architectures = config.get("architectures", [])
            for arch in architectures:
                if arch in self._CHAT_TEMPLATES:
                    logger.info("Chat template 감지됨: %s", arch)
                    return self._CHAT_TEMPLATES[arch]
            logger.info(
                "알 수 없는 아키텍처(%s) — chat template 생략 (Ollama 자동 감지에 의존)",
                architectures,
            )
        except Exception as e:
            logger.warning("config.json 파싱 실패: %s", e)

        return None

    def _detect_stop_token(self, model_dir: Path) -> str | None:
        """모델 아키텍처에 맞는 stop 토큰을 반환합니다."""
        import json

        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for arch in config.get("architectures", []):
                if arch in self._STOP_TOKENS:
                    return self._STOP_TOKENS[arch]
        except Exception:
            pass

        return None

    def create_model(
        self,
        modelfile_path: str | Path,
        model_name_override: str | None = None,
    ) -> bool:
        """Modelfile에서 Ollama 모델을 생성합니다.
        
        `ollama create` 명령을 실행하여 모델을 가져옵니다.
        
        인자:
            modelfile_path: Modelfile의 경로
            model_name_override: 지정 시 설정 대신 이 이름으로 모델을 생성합니다
        
        반환값:
            성공하면 True, 그렇지 않으면 False
        """
        from rich.console import Console
        
        console = Console()
        modelfile_path = Path(modelfile_path)
        target_name = model_name_override or self.model_name
        
        logger.info("Ollama 모델 생성 중: %s", target_name)
        
        try:
            with console.status(f"[bold blue]Ollama 모델 생성 중: {target_name}[/bold blue]"):
                result = subprocess.run(
                    ["ollama", "create", target_name, "-f", str(modelfile_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            
            if result.returncode == 0:
                logger.info("✓ Ollama 모델 생성됨: %s", target_name)
                logger.info("✓ 모델 준비 완료! 실행: ollama run %s", target_name)
            else:
                error_detail = (result.stderr or result.stdout or "").strip()
                logger.warning(
                    "Ollama 모델 생성 실패 (종료 코드 %d): %s",
                    result.returncode,
                    error_detail or "(상세 오류 없음)",
                )
                if "unsupported architecture" in error_detail:
                    logger.warning(
                        "이 모델 아키텍처는 Ollama safetensors 변환을 아직 지원하지 않습니다. "
                        "GGUF 변환 후 재시도하거나 지원되는 아키텍처(Qwen2, Llama, Gemma 등)를 사용하세요."
                    )
            return result.returncode == 0
                
        except FileNotFoundError:
            logger.warning("ollama 명령을 찾을 수 없음")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("ollama create 명령 타임아웃")
            return False
        except Exception as e:
            logger.warning("ollama create 실행 중 오류: %s", e)
            return False
    
    def export(
        self,
        model_dir: str | Path,
        output_dir: str | Path | None = None,
        model_name_override: str | None = None,
    ) -> Path:
        """모델을 Ollama 형식으로 내보냅니다.
        
        Modelfile을 생성하고 선택적으로 모델을 생성하는 주요 진입점입니다.
        
        인자:
            model_dir: HuggingFace 모델 디렉토리의 경로
            output_dir: Modelfile의 출력 디렉토리 (기본값: model_dir)
            model_name_override: 지정 시 설정 대신 이 이름으로 모델을 생성합니다
        
        반환값:
            생성된 Modelfile의 경로
        """
        model_dir = Path(model_dir)
        target_name = model_name_override or self.model_name
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            modelfile_path = output_dir / "Modelfile"
        else:
            modelfile_path = None
        
        modelfile_path = self.generate_modelfile(model_dir, modelfile_path)
        
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ollama_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ollama_available = False
        
        if ollama_available:
            logger.info("Ollama 감지됨, 모델 생성 시도 중...")
            success = self.create_model(modelfile_path, model_name_override=model_name_override)
            if not success:
                logger.info("수동으로 가져오려면 실행: ollama create %s -f %s", target_name, modelfile_path)
        else:
            logger.info("Ollama를 감지하지 못했습니다. 수동으로 가져오려면:")
            logger.info("  1. https://ollama.ai에서 Ollama 설치")
            logger.info("  2. 실행: ollama create %s -f %s", target_name, modelfile_path)
        
        return modelfile_path
