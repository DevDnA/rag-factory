"""FileBackedSessionStore 테스트 — 디스크 영속화와 TTL 정책을 검증합니다."""

from __future__ import annotations

import time

import pytest

from slm_factory.rag.agent.session import Message
from slm_factory.rag.agent.state import FileBackedSessionStore, SessionRecord


@pytest.fixture
def store(tmp_path):
    return FileBackedSessionStore(
        base_dir=tmp_path / "sessions",
        ttl=3600,
        max_turns=5,
        max_sessions=10,
    )


class TestBasicOperations:
    """기본 CRUD 동작."""

    def test_디렉터리_자동_생성(self, tmp_path):
        target = tmp_path / "nested" / "sessions"
        FileBackedSessionStore(base_dir=target)
        assert target.is_dir()

    def test_세션_생성_파일_저장(self, store, tmp_path):
        sid = store.create_session()
        assert sid
        assert (tmp_path / "sessions" / f"{sid}.json").exists()

    def test_get_or_create_새_세션(self, store):
        sid, msgs = store.get_or_create(None)
        assert sid
        assert msgs == []

    def test_get_or_create_기존_세션(self, store):
        sid = store.create_session()
        store.add_message(sid, Message(role="user", content="hi"))
        sid2, msgs = store.get_or_create(sid)
        assert sid2 == sid
        assert len(msgs) == 1
        assert msgs[0].content == "hi"

    def test_get_or_create_없는_ID는_새로_생성(self, store):
        sid, msgs = store.get_or_create("nonexistent_xxx")
        assert sid != "nonexistent_xxx"
        assert msgs == []


class TestPersistence:
    """서버 재시작 시뮬레이션 — 새 인스턴스에서 세션이 이어져야 합니다."""

    def test_재시작_후_세션_복원(self, tmp_path):
        base = tmp_path / "sessions"
        store1 = FileBackedSessionStore(base_dir=base, ttl=3600, max_turns=5)
        sid = store1.create_session()
        store1.add_message(sid, Message(role="user", content="첫 질문"))
        store1.add_message(sid, Message(role="assistant", content="첫 답변"))

        store2 = FileBackedSessionStore(base_dir=base, ttl=3600, max_turns=5)
        sid2, msgs = store2.get_or_create(sid)
        assert sid2 == sid
        assert len(msgs) == 2
        assert msgs[0].content == "첫 질문"
        assert msgs[1].content == "첫 답변"

    def test_format_history_도_복원됨(self, tmp_path):
        base = tmp_path / "sessions"
        store1 = FileBackedSessionStore(base_dir=base)
        sid = store1.create_session()
        store1.add_message(sid, Message(role="user", content="안녕"))
        store1.add_message(sid, Message(role="assistant", content="안녕하세요"))

        store2 = FileBackedSessionStore(base_dir=base)
        history = store2.format_history(sid)
        assert "[이전 대화]" in history
        assert "사용자: 안녕" in history
        assert "어시스턴트: 안녕하세요" in history


class TestLimits:
    """max_turns·max_sessions 제한."""

    def test_max_turns_초과시_오래된_메시지_제거(self, store):
        sid = store.create_session()
        for i in range(15):
            store.add_message(sid, Message(role="user", content=f"u{i}"))
            store.add_message(sid, Message(role="assistant", content=f"a{i}"))
        _, msgs = store.get_or_create(sid)
        # max_turns(5) * 2 = 10
        assert len(msgs) == 10
        # 최신 메시지만 유지
        assert msgs[-1].content == "a14"

    def test_max_sessions_초과시_최고령_세션_퇴거(self, tmp_path):
        store = FileBackedSessionStore(
            base_dir=tmp_path / "s",
            ttl=3600,
            max_turns=5,
            max_sessions=3,
        )
        sids = []
        for _ in range(3):
            sid = store.create_session()
            sids.append(sid)
            time.sleep(0.01)

        # 4번째 세션 — 가장 오래된 sids[0]이 퇴거되어야 함
        new_sid = store.create_session()
        assert store.active_count == 3

        sid2, msgs = store.get_or_create(sids[0])
        # 퇴거되었으므로 새 세션이 생성됨
        assert sid2 != sids[0]

        # 최신 세션은 남아 있음
        sid_latest, _ = store.get_or_create(new_sid)
        assert sid_latest == new_sid


class TestCleanup:
    """TTL 기반 만료 정리."""

    def test_cleanup_expired_만료_세션_제거(self, tmp_path):
        store = FileBackedSessionStore(base_dir=tmp_path / "s", ttl=0)
        sid = store.create_session()
        store.add_message(sid, Message(role="user", content="x"))
        time.sleep(0.02)
        removed = store.cleanup_expired()
        assert removed == 1
        assert store.active_count == 0

    def test_cleanup_활성_세션_유지(self, store):
        store.create_session()
        removed = store.cleanup_expired()
        assert removed == 0
        assert store.active_count == 1


class TestSerialization:
    """SessionRecord 직렬화 라운드트립."""

    def test_to_from_dict_라운드트립(self):
        rec = SessionRecord(
            session_id="abc",
            messages=[
                Message(role="user", content="q", timestamp=1.0),
                Message(role="assistant", content="a", timestamp=2.0),
            ],
            last_access=3.0,
        )
        data = rec.to_dict()
        restored = SessionRecord.from_dict(data)
        assert restored.session_id == "abc"
        assert restored.last_access == 3.0
        assert len(restored.messages) == 2
        assert restored.messages[0].role == "user"
        assert restored.messages[1].content == "a"


class TestCorruption:
    """손상된 세션 파일에 대한 방어적 처리."""

    def test_손상된_JSON은_None_반환(self, tmp_path):
        base = tmp_path / "s"
        store = FileBackedSessionStore(base_dir=base)
        (base / "broken.json").write_text("{not json", encoding="utf-8")

        # 손상된 ID로 조회 시 새 세션 생성 (크래시 없음)
        sid, msgs = store.get_or_create("broken")
        assert sid != "broken"
        assert msgs == []


class TestLastSources:
    """Phase 3-a — last_sources 영속화."""

    def test_기본값은_빈_목록(self, store):
        sid = store.create_session()
        assert store.get_last_sources(sid) == []

    def test_set_후_get(self, store):
        sid = store.create_session()
        srcs = [{"doc_id": "d1", "score": 0.9, "content": "c"}]
        store.set_last_sources(sid, srcs)
        assert store.get_last_sources(sid) == srcs

    def test_재시작_후에도_복원(self, tmp_path):
        base = tmp_path / "s"
        store1 = FileBackedSessionStore(base_dir=base)
        sid = store1.create_session()
        store1.set_last_sources(sid, [{"doc_id": "d1", "content": "c"}])

        store2 = FileBackedSessionStore(base_dir=base)
        assert store2.get_last_sources(sid) == [{"doc_id": "d1", "content": "c"}]

    def test_없는_세션에_set은_무시(self, store):
        store.set_last_sources("nonexistent", [{"doc_id": "x"}])
        assert store.get_last_sources("nonexistent") == []

    def test_dict이_아닌_entry는_필터링(self, store):
        sid = store.create_session()
        store.set_last_sources(sid, [{"doc_id": "ok"}, "not a dict", 42, None])
        restored = store.get_last_sources(sid)
        assert restored == [{"doc_id": "ok"}]

    def test_손상된_JSON_last_sources도_안전(self, tmp_path):
        base = tmp_path / "s"
        store = FileBackedSessionStore(base_dir=base)
        sid = store.create_session()
        # 수동으로 손상된 last_sources 삽입
        path = base / f"{sid}.json"
        import json as _json
        data = _json.loads(path.read_text())
        data["last_sources"] = "not a list"
        path.write_text(_json.dumps(data))

        assert store.get_last_sources(sid) == []
