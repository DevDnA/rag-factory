"""증분 학습을 위한 문서 변경 추적기입니다."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SLMConfig
    from .models import QAPair

from .utils import get_logger

logger = get_logger("incremental")


class IncrementalTracker:
    """문서 해시 기반 증분 변경 추적기입니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self._pending_hashes: dict[str, str] | None = None
        self._pending_hash_file: Path | None = None
    def compute_document_hashes(
        self, doc_dir: Path, formats: list[str],
    ) -> dict[str, str]:
        """디렉토리 내 문서들의 SHA-256 해시를 계산합니다."""
        from .utils import compute_file_hash

        extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in formats
        ]
        hashes: dict[str, str] = {}
        doc_path = Path(doc_dir)
        if not doc_path.is_dir():
            return hashes
        for f in sorted(doc_path.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                hashes[f.name] = compute_file_hash(f)
        return hashes

    def load_saved_hashes(self, hash_file: Path) -> dict[str, str]:
        """저장된 해시 파일을 로드합니다."""
        import json

        path = Path(hash_file)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "해시 파일이 손상되어 초기화합니다 (전체 문서 재처리 필요): %s",
                path,
            )
            return {}

    def save_hashes(self, hashes: dict[str, str], hash_file: Path) -> None:
        """해시를 JSON 파일로 저장합니다."""
        import json

        path = Path(hash_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def detect_changes(
        self, current: dict[str, str], saved: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        """현재와 저장된 해시를 비교하여 변경사항을 감지합니다."""
        current_keys = set(current)
        saved_keys = set(saved)

        new_files = sorted(current_keys - saved_keys)
        deleted_files = sorted(saved_keys - current_keys)
        modified_files = sorted(
            k for k in current_keys & saved_keys if current[k] != saved[k]
        )

        return new_files, modified_files, deleted_files

    def merge_qa_pairs(
        self,
        existing: list[QAPair],
        new_pairs: list[QAPair],
        strategy: str,
    ) -> list[QAPair]:
        """기존 QA 쌍과 새 QA 쌍을 병합합니다.

        append 전략에서는 새 쌍을 우선하여 업데이트된 답변이 반영되도록 합니다.
        중복 판단 키는 (question, source_doc) 조합입니다.
        """
        if strategy == "replace":
            merged = list(new_pairs)
        else:
            # 새 쌍을 먼저 배치하여 업데이트 우선
            merged = list(new_pairs) + list(existing)

        seen: set[tuple[str, str]] = set()
        deduped: list[QAPair] = []
        for pair in merged:
            key = (pair.question, pair.source_doc)
            if key not in seen:
                seen.add(key)
                deduped.append(pair)

        return deduped

    def get_changed_files(self, doc_dir: Path) -> list[Path]:
        """변경된 파일 목록을 반환합니다.

        해시는 즉시 저장하지 않고, :meth:`commit_hashes`를 호출해야
        저장됩니다. 이로써 파이프라인 처리 중 크래시 시 해시가
        이미 업데이트되어 변경 파일이 누락되는 문제를 방지합니다.
        """
        doc_path = Path(doc_dir)
        formats = self.config.parsing.formats
        hash_file = Path(self.config.paths.output) / self.config.incremental.hash_file

        current = self.compute_document_hashes(doc_path, formats)
        saved = self.load_saved_hashes(hash_file)
        new_files, modified_files, _deleted = self.detect_changes(current, saved)

        # 해시를 바로 저장하지 않고 보류 — commit_hashes()에서 저장
        self._pending_hashes = current
        self._pending_hash_file = hash_file

        changed = [doc_path / name for name in new_files + modified_files]
        logger.info(
            "Incremental: %d new, %d modified, %d deleted",
            len(new_files), len(modified_files), len(_deleted),
        )
        return changed

    def commit_hashes(self) -> None:
        """보류 중인 해시를 파일로 저장합니다.

        :meth:`get_changed_files` 호출 후 실제 처리가 완료된 시점에
        호출하여 해시를 확정합니다.
        """
        if self._pending_hashes is not None and self._pending_hash_file is not None:
            self.save_hashes(self._pending_hashes, self._pending_hash_file)
            self._pending_hashes = None
            self._pending_hash_file = None
