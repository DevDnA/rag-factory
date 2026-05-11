"""파일 기반 영속 세션 저장소 — 서버 재시작 시에도 대화 내역을 유지합니다.

기존 ``SessionManager`` (in-memory)와 동일한 API를 노출하므로 서버 코드가
교체 없이 양쪽을 모두 사용할 수 있습니다. 활성화는
``config.rag.agent.persist_sessions``로 제어합니다.

설계 노트
---------
- 세션당 하나의 JSON 파일을 사용하여 동시 세션 간 파일 경합을 최소화합니다.
- atomic write는 ``os.replace``로 구현하여 쓰기 중 장애가 나도 파일이 반쯤
  쓰여진 상태로 남지 않습니다.
- 단일 프로세스 가정 — 여러 uvicorn 워커에서 공유하려면 후속 이터레이션에서
  파일 락 또는 외부 저장소로 업그레이드해야 합니다.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ...utils import get_logger
from .session import Message

logger = get_logger("rag.agent.state")


@dataclass
class SessionRecord:
    """디스크에 직렬화되는 단일 세션 레코드."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)
    last_sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "last_access": self.last_access,
            "messages": [asdict(m) for m in self.messages],
            "last_sources": list(self.last_sources),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionRecord":
        msgs = [
            Message(
                role=m.get("role", ""),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", time.time()),
            )
            for m in data.get("messages", [])
        ]
        raw_sources = data.get("last_sources") or []
        if not isinstance(raw_sources, list):
            raw_sources = []
        return cls(
            session_id=data["session_id"],
            messages=msgs,
            last_access=data.get("last_access", time.time()),
            last_sources=[s for s in raw_sources if isinstance(s, dict)],
        )


class FileBackedSessionStore:
    """파일 시스템에 세션을 영속화하는 저장소.

    ``SessionManager``와 동일한 공개 API (``create_session``,
    ``get_or_create``, ``add_message``, ``format_history``,
    ``cleanup_expired``, ``active_count``)를 제공합니다.

    Parameters
    ----------
    base_dir:
        세션 JSON 파일들이 저장될 디렉터리. 없으면 자동 생성됩니다.
    ttl:
        세션 TTL(초). 마지막 접근 시점 기준으로 만료를 판정합니다.
    max_turns:
        세션당 보존할 최대 턴 수 (user+assistant 쌍 기준).
    max_sessions:
        동시에 유지할 수 있는 최대 세션 수. 초과 시 최고령 세션을 퇴거합니다.
    """

    def __init__(
        self,
        base_dir: str | os.PathLike[str],
        ttl: int = 3600,
        max_turns: int = 20,
        max_sessions: int = 1000,
    ) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl
        self._max_turns = max_turns
        self._max_sessions = max_sessions

    # ------------------------------------------------------------------
    # 내부 I/O
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _load(self, session_id: str) -> SessionRecord | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionRecord.from_dict(data)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("세션 로드 실패 (%s): %s — 무시합니다", session_id, exc)
            return None

    def _save(self, record: SessionRecord) -> None:
        path = self._path(record.session_id)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(record.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("세션 저장 실패 (%s): %s", record.session_id, exc)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _delete(self, session_id: str) -> None:
        path = self._path(session_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("세션 파일 삭제 실패 (%s): %s", session_id, exc)

    def _iter_records(self) -> list[SessionRecord]:
        records: list[SessionRecord] = []
        for path in self._dir.glob("*.json"):
            sid = path.stem
            rec = self._load(sid)
            if rec is not None:
                records.append(rec)
        return records

    # ------------------------------------------------------------------
    # Public API (SessionManager와 호환)
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """새 세션을 생성하고 ID를 반환합니다."""
        records = self._iter_records()
        if len(records) >= self._max_sessions:
            self.cleanup_expired()
            records = self._iter_records()
            if len(records) >= self._max_sessions:
                oldest = min(records, key=lambda r: r.last_access)
                self._delete(oldest.session_id)
                logger.debug(
                    "세션 수 제한(%d) 도달 — 최고령 세션 '%s' 퇴거",
                    self._max_sessions,
                    oldest.session_id,
                )

        session_id = uuid.uuid4().hex[:12]
        self._save(SessionRecord(session_id=session_id))
        return session_id

    def get_or_create(
        self, session_id: str | None
    ) -> tuple[str, list[Message]]:
        """세션을 조회하거나 없으면 새로 생성합니다."""
        if session_id:
            record = self._load(session_id)
            if record is not None:
                record.last_access = time.time()
                self._save(record)
                return session_id, list(record.messages)

        new_id = self.create_session()
        return new_id, []

    def add_message(self, session_id: str, message: Message) -> None:
        """세션에 메시지를 추가합니다. TTL·max_turns 규칙을 적용합니다."""
        record = self._load(session_id) or SessionRecord(session_id=session_id)
        record.messages.append(message)
        record.last_access = time.time()

        cap = self._max_turns * 2
        if len(record.messages) > cap:
            excess = len(record.messages) - cap
            record.messages = record.messages[excess:]

        self._save(record)

    def format_history(self, session_id: str) -> str:
        """프롬프트에 삽입할 대화 내역 텍스트를 생성합니다."""
        record = self._load(session_id)
        if record is None or not record.messages:
            return ""

        lines: list[str] = ["[이전 대화]"]
        for msg in record.messages:
            if msg.role == "user":
                lines.append(f"사용자: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"어시스턴트: {msg.content}")
        lines.append("")
        return "\n".join(lines)

    def cleanup_expired(self) -> int:
        """TTL이 만료된 세션 파일을 정리합니다.

        ``_save``가 모든 메시지/last_sources 변경 시점마다 호출되므로
        파일 mtime ≈ ``last_access``가 보장됩니다. 따라서 JSON을 모두 로드해
        ``last_access``를 비교하는 대신 ``Path.stat().st_mtime``으로 빠르게
        만료 여부를 판정해 I/O 비용을 절감합니다.
        """
        now = time.time()
        removed = 0
        for path in self._dir.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError as exc:
                logger.debug("세션 mtime 조회 실패 (%s): %s", path.name, exc)
                continue
            if now - mtime > self._ttl:
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError as exc:
                    logger.debug("만료 세션 삭제 실패 (%s): %s", path.name, exc)
        if removed:
            logger.debug("만료 세션 %d개 정리 완료", removed)
        return removed

    def compress_old_turns(
        self, session_id: str, keep_recent: int, summary_text: str
    ) -> int:
        """오래된 메시지를 요약 메시지 하나로 교체 (FileBackedSessionStore용)."""
        record = self._load(session_id)
        if record is None:
            return 0
        if len(record.messages) <= keep_recent:
            return 0
        removed = len(record.messages) - keep_recent
        summary_msg = Message(
            role="assistant",
            content=f"[이전 대화 요약] {summary_text}",
        )
        record.messages = [summary_msg] + record.messages[-keep_recent:]
        record.last_access = time.time()
        self._save(record)
        return removed

    def set_last_sources(self, session_id: str, sources: list[dict]) -> None:
        """세션에 "가장 최근 참조 문서" 목록을 저장합니다."""
        record = self._load(session_id)
        if record is None:
            return
        record.last_sources = [s for s in sources if isinstance(s, dict)]
        record.last_access = time.time()
        self._save(record)

    def get_last_sources(self, session_id: str) -> list[dict]:
        """세션의 "가장 최근 참조 문서" 목록을 반환합니다."""
        record = self._load(session_id)
        if record is None:
            return []
        return list(record.last_sources)

    @property
    def active_count(self) -> int:
        """현재 활성 세션 수."""
        return len(list(self._dir.glob("*.json")))


__all__ = ["FileBackedSessionStore", "SessionRecord"]
