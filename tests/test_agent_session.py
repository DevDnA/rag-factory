"""세션 관리자 테스트."""

from __future__ import annotations

import time

from slm_factory.rag.agent.session import Message, SessionManager


class TestSessionManager:
    """SessionManager 클래스 테스트."""

    def test_세션_생성(self):
        sm = SessionManager(ttl=60, max_turns=10)
        sid = sm.create_session()
        assert sid
        assert sm.active_count == 1

    def test_get_or_create_새_세션(self):
        sm = SessionManager()
        sid, msgs = sm.get_or_create(None)
        assert sid
        assert msgs == []
        assert sm.active_count == 1

    def test_get_or_create_기존_세션(self):
        sm = SessionManager()
        sid1 = sm.create_session()
        sm.add_message(sid1, Message(role="user", content="hello"))
        sid2, msgs = sm.get_or_create(sid1)
        assert sid2 == sid1
        assert len(msgs) == 1

    def test_get_or_create_없는_세션_새로_생성(self):
        sm = SessionManager()
        sid, msgs = sm.get_or_create("nonexistent_id")
        assert sid != "nonexistent_id"
        assert msgs == []

    def test_메시지_추가(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, Message(role="user", content="질문입니다"))
        sm.add_message(sid, Message(role="assistant", content="답변입니다"))
        _, msgs = sm.get_or_create(sid)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_max_turns_초과시_오래된_메시지_제거(self):
        sm = SessionManager(max_turns=2)
        sid = sm.create_session()
        # max_turns=2이면 최대 4개 메시지 (user+assistant 쌍)
        for i in range(10):
            sm.add_message(sid, Message(role="user", content=f"q{i}"))
            sm.add_message(sid, Message(role="assistant", content=f"a{i}"))
        _, msgs = sm.get_or_create(sid)
        assert len(msgs) == 4  # max_turns * 2

    def test_format_history(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, Message(role="user", content="안녕하세요"))
        sm.add_message(sid, Message(role="assistant", content="안녕하세요!"))
        history = sm.format_history(sid)
        assert "사용자: 안녕하세요" in history
        assert "어시스턴트: 안녕하세요!" in history
        assert "[이전 대화]" in history

    def test_format_history_빈_세션(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.format_history(sid) == ""

    def test_format_history_없는_세션(self):
        sm = SessionManager()
        assert sm.format_history("nonexistent") == ""

    def test_cleanup_expired(self):
        sm = SessionManager(ttl=0)  # 즉시 만료
        sid = sm.create_session()
        sm.add_message(sid, Message(role="user", content="test"))
        time.sleep(0.01)
        removed = sm.cleanup_expired()
        assert removed == 1
        assert sm.active_count == 0

    def test_cleanup_활성_세션_유지(self):
        sm = SessionManager(ttl=3600)
        sm.create_session()
        removed = sm.cleanup_expired()
        assert removed == 0
        assert sm.active_count == 1

    def test_여러_세션_관리(self):
        sm = SessionManager()
        sid1 = sm.create_session()
        sid2 = sm.create_session()
        assert sid1 != sid2
        assert sm.active_count == 2
        sm.add_message(sid1, Message(role="user", content="a"))
        sm.add_message(sid2, Message(role="user", content="b"))
        _, msgs1 = sm.get_or_create(sid1)
        _, msgs2 = sm.get_or_create(sid2)
        assert msgs1[0].content == "a"
        assert msgs2[0].content == "b"


class TestLastSources:
    """Phase 3-a — set/get_last_sources API."""

    def test_기본값은_빈_목록(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.get_last_sources(sid) == []

    def test_set_후_get(self):
        sm = SessionManager()
        sid = sm.create_session()
        srcs = [{"doc_id": "d1", "score": 0.9, "content": "c"}]
        sm.set_last_sources(sid, srcs)
        assert sm.get_last_sources(sid) == srcs

    def test_없는_세션에_set은_무시(self):
        sm = SessionManager()
        sm.set_last_sources("nonexistent", [{"doc_id": "x"}])
        assert sm.get_last_sources("nonexistent") == []

    def test_set은_복사본_저장(self):
        sm = SessionManager()
        sid = sm.create_session()
        srcs = [{"doc_id": "d1"}]
        sm.set_last_sources(sid, srcs)
        srcs.append({"doc_id": "d2"})  # 외부에서 변경해도
        assert len(sm.get_last_sources(sid)) == 1  # 내부는 영향 없음

    def test_cleanup_expired시_last_sources도_제거(self):
        sm = SessionManager(ttl=0)
        sid = sm.create_session()
        sm.set_last_sources(sid, [{"doc_id": "d1"}])
        time.sleep(0.01)
        sm.cleanup_expired()
        assert sm.get_last_sources(sid) == []
