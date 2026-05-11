"""진화 히스토리 관리 — 버전 추적, 품질 게이트, 이전 버전 정리."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import EvolveConfig, SLMConfig
    from .models import CompareResult

from .utils import get_logger

logger = get_logger("evolve_history")


class EvolveHistory:
    """evolve_history.json 기반 진화 상태 관리자입니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.evolve_config: EvolveConfig = config.evolve
        self.history_path = (
            Path(config.paths.output) / self.evolve_config.history_file
        )

    def load(self) -> dict[str, Any]:
        if not self.history_path.is_file():
            return {"versions": [], "current": None}
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "진화 히스토리 파일이 손상되어 초기화합니다: %s",
                self.history_path,
            )
            return {"versions": [], "current": None}

    def save(self, data: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_first_run(self) -> bool:
        history = self.load()
        return len(history.get("versions", [])) == 0

    def get_current_model_name(self) -> str | None:
        history = self.load()
        current = history.get("current")
        if current is None:
            return None
        for entry in history.get("versions", []):
            if entry.get("version") == current:
                return entry.get("model_name")
        return None

    def generate_version_name(self) -> str:
        now = datetime.now(tz=timezone.utc)
        date_str = now.strftime("%Y%m%d")
        base_version = f"v{date_str}"

        history = self.load()
        existing = [
            e["version"]
            for e in history.get("versions", [])
            if e.get("version", "").startswith(base_version)
        ]

        if not existing:
            return base_version

        suffix = 2
        while f"{base_version}-{suffix}" in existing:
            suffix += 1
        return f"{base_version}-{suffix}"

    def generate_model_name(self, version: str) -> str:
        base_name = self.config.export.ollama.model_name
        return f"{base_name}-{version}"

    def check_quality_gate(
        self,
        results: list[CompareResult],
    ) -> tuple[bool, dict[str, float]]:
        metric = self.evolve_config.gate_metric
        min_improvement = self.evolve_config.gate_min_improvement

        if not results:
            return False, {}

        base_key = f"base_{metric}"
        ft_key = f"finetuned_{metric}"

        base_vals = [r.scores[base_key] for r in results if base_key in r.scores]
        ft_vals = [r.scores[ft_key] for r in results if ft_key in r.scores]

        if not base_vals or not ft_vals:
            logger.warning(
                "메트릭 '%s'의 점수를 찾을 수 없습니다", metric,
            )
            return False, {}

        base_avg = sum(base_vals) / len(base_vals)
        ft_avg = sum(ft_vals) / len(ft_vals)

        if base_avg > 0:
            improvement_pct = (ft_avg - base_avg) / base_avg * 100
        else:
            improvement_pct = 100.0 if ft_avg > 0 else 0.0

        scores = {
            "base_avg": round(base_avg, 4),
            "finetuned_avg": round(ft_avg, 4),
            "improvement_pct": round(improvement_pct, 2),
        }

        passed = improvement_pct >= min_improvement
        return passed, scores

    def record_version(
        self,
        version: str,
        model_name: str,
        scores: dict[str, float] | None = None,
        qa_count: int = 0,
        *,
        promoted: bool = False,
    ) -> None:
        history = self.load()

        entry: dict[str, Any] = {
            "version": version,
            "model_name": model_name,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "qa_count": qa_count,
            "promoted": promoted,
        }
        if scores:
            entry["scores"] = scores

        history.setdefault("versions", []).append(entry)

        if promoted:
            history["current"] = version

        self.save(history)
        logger.info("버전 기록: %s (promoted=%s)", version, promoted)

    def cleanup_old_versions(self) -> list[str]:
        keep = self.evolve_config.keep_previous_versions
        if keep <= 0:
            return []

        history = self.load()
        versions = history.get("versions", [])
        current = history.get("current")

        promoted = [v for v in versions if v.get("promoted")]
        if len(promoted) <= keep:
            return []

        to_remove = promoted[: len(promoted) - keep]
        removed_versions: set[str] = set()

        removed_names: list[str] = []
        for entry in to_remove:
            if entry.get("version") == current:
                continue
            model_name = entry.get("model_name", "")
            if model_name:
                self._ollama_rm(model_name)
                removed_names.append(model_name)
                removed_versions.add(entry.get("version", ""))

        # 히스토리 JSON에서도 삭제된 버전 제거
        if removed_versions:
            history["versions"] = [
                v for v in versions
                if v.get("version") not in removed_versions
            ]

        # 비승격 버전(품질 게이트 실패)도 히스토리에서 정리
        non_promoted = [
            v for v in history.get("versions", [])
            if not v.get("promoted") and v.get("version") != current
        ]
        if non_promoted:
            non_promoted_versions = {v.get("version") for v in non_promoted}
            history["versions"] = [
                v for v in history.get("versions", [])
                if v.get("version") not in non_promoted_versions
            ]
            removed_versions.update(non_promoted_versions)

        if removed_versions:
            self.save(history)

        return removed_names

    @staticmethod
    def _ollama_rm(model_name: str) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("이전 모델 삭제: %s", model_name)
                return True
            logger.warning("모델 삭제 실패: %s (%s)", model_name, result.stderr.strip())
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("ollama rm 실행 불가: %s", model_name)
            return False
