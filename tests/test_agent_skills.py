"""Skills 시스템 테스트 — trigger 매칭, 로딩, registry, orchestrator 통합."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from slm_factory.rag.agent.skills import (
    Skill,
    SkillRegistry,
    load_skills_from_dir)


class TestSkillMatching:
    def test_plain_trigger(self):
        s = Skill(name="legal", triggers=("조항", "판례"))
        assert s.matches("제15조의 조항은?") is True
        assert s.matches("판례 검색") is True
        assert s.matches("날씨 어때?") is False

    def test_대소문자_무시(self):
        s = Skill(name="api", triggers=("API",))
        assert s.matches("api spec") is True
        assert s.matches("Api reference") is True

    def test_regex_trigger(self):
        s = Skill(name="num", triggers=("regex:\\d{3}-\\d{4}",))
        assert s.matches("전화번호 010-1234") is True
        assert s.matches("전화번호 일천이삼사") is False

    def test_빈_triggers(self):
        s = Skill(name="x", triggers=())
        assert s.matches("무엇이든") is False

    def test_잘못된_regex는_무시(self):
        s = Skill(name="x", triggers=("regex:[invalid",))
        assert s.matches("anything") is False  # 예외 없이 False


class TestLoadSkills:
    def test_존재하지_않는_디렉터리(self, tmp_path):
        skills = load_skills_from_dir(tmp_path / "no-such")
        assert skills == []

    def test_빈_디렉터리(self, tmp_path):
        skills = load_skills_from_dir(tmp_path)
        assert skills == []

    def test_단일_skill_로드(self, tmp_path):
        (tmp_path / "legal.yaml").write_text(
            "name: legal\n"
            "description: 법률 지식\n"
            "triggers: [조항, 판례]\n"
            "prompt_addon: |\n  법률 답변 규칙: 조항 인용 필수\n"
            "priority: 5\n",
            encoding="utf-8")
        skills = load_skills_from_dir(tmp_path)
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "legal"
        assert "조항" in s.triggers
        assert "조항 인용" in s.prompt_addon
        assert s.priority == 5

    def test_여러_skill_로드(self, tmp_path):
        (tmp_path / "a.yaml").write_text("name: a\ntriggers: [foo]\n", encoding="utf-8")
        (tmp_path / "b.yml").write_text("name: b\ntriggers: [bar]\n", encoding="utf-8")
        skills = load_skills_from_dir(tmp_path)
        names = {s.name for s in skills}
        assert names == {"a", "b"}

    def test_잘못된_YAML은_건너뜀(self, tmp_path):
        (tmp_path / "good.yaml").write_text("name: ok\ntriggers: [x]\n", encoding="utf-8")
        (tmp_path / "bad.yaml").write_text("{not json: [oops", encoding="utf-8")
        skills = load_skills_from_dir(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "ok"

    def test_name_없는_skill은_건너뜀(self, tmp_path):
        (tmp_path / "noname.yaml").write_text("triggers: [x]\n", encoding="utf-8")
        skills = load_skills_from_dir(tmp_path)
        assert skills == []

    def test_하위디렉터리도_스캔(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.yaml").write_text("name: nested\ntriggers: [x]\n", encoding="utf-8")
        skills = load_skills_from_dir(tmp_path)
        assert len(skills) == 1


class TestSkillRegistry:
    def test_select_for_query(self):
        reg = SkillRegistry([
            Skill(name="legal", triggers=("조항"), prompt_addon="법률 규칙"),
            Skill(name="finance", triggers=("금리"), prompt_addon="금융 규칙"),
        ])
        matches = reg.select_for_query("제15조 조항")
        assert len(matches) == 1
        assert matches[0].name == "legal"

    def test_priority_내림차순(self):
        reg = SkillRegistry([
            Skill(name="low", triggers=("x"), priority=1),
            Skill(name="high", triggers=("x"), priority=10),
        ])
        matches = reg.select_for_query("x 있음")
        assert matches[0].name == "high"
        assert matches[1].name == "low"

    def test_limit(self):
        reg = SkillRegistry([
            Skill(name=f"s{i}", triggers=("x"), priority=i)
            for i in range(5)
        ])
        matches = reg.select_for_query("x", limit=2)
        assert len(matches) == 2
        assert [s.name for s in matches] == ["s4", "s3"]

    def test_format_addons(self):
        skills = [
            Skill(name="legal", prompt_addon="법률 규칙"),
            Skill(name="finance", prompt_addon="금융 규칙"),
        ]
        formatted = SkillRegistry.format_addons(skills)
        assert "[도메인 스킬]" in formatted
        assert "## legal" in formatted
        assert "법률 규칙" in formatted
        assert "## finance" in formatted

    def test_format_addons_빈_addon_제외(self):
        skills = [
            Skill(name="legal", prompt_addon="법률 규칙"),
            Skill(name="empty", prompt_addon="   "),
        ]
        formatted = SkillRegistry.format_addons(skills)
        assert "## legal" in formatted
        assert "## empty" not in formatted


class TestOrchestratorIntegration:
    """orchestrator가 skill addon을 synthesis prompt에 주입하는지."""

    @pytest.mark.asyncio
    async def test_skill_활성시_prompt에_주입(self, tmp_path, monkeypatch):
        # skill 디렉터리 준비
        (tmp_path / "legal.yaml").write_text(
            "name: legal\n"
            "triggers: [조항, 법]\n"
            "prompt_addon: |\n  LEGAL_DOMAIN_RULE_INJECTED\n",
            encoding="utf-8")

        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_orchestrator,
            _make_plan,
            _FakeToolResult,
            _collect)

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])

        # config에 skills_enabled=True + skills_dir=tmp_path 설정
        config = fixtures.app_state  # reuse namespace
        orch = _make_orchestrator(
            planner_enabled=True,
            app_state=fixtures.app_state)
        # orchestrator 직접 생성 (skills_enabled 주입 필요)
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.router import QueryRouter

        config_ns = SimpleNamespace(
            rag=SimpleNamespace(
                agent=SimpleNamespace(
                    enabled=True,
                    max_iterations=3,
                    stream_reasoning=False,
                    planner_enabled=True,
                    verifier_enabled=True,
                    verifier_max_repairs=1,
                    legacy_fallback_enabled=True,
                    session_source_reuse=False,
                    session_source_reuse_limit=5,
                    parallel_steps=False,
                    clarifier_enabled=False,
                    clarifier_max_questions=2,
                    personas_enabled=False,
                    native_thinking=False,
                    skills_enabled=True,
                    skills_dir=str(tmp_path)),
                request_timeout=60.0))

        async def _simple_fn(q):
            yield {"type": "token", "content": "simple"}
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=QueryRouter(agent_enabled=True),
            app_state=fixtures.app_state,
            config=config_ns,
            ollama_model="test",
            api_base="http://localhost:11434",
            rag_max_tokens=-1,
            simple_stream_fn=_simple_fn)

        # 조항 키워드가 있는 질의 → agent 경로
        await _collect(orch.handle_agent("제15조 조항 관련 질의"))

        prompt = fixtures.app_state.http_client.stream.call_args.kwargs["json"]["prompt"]
        assert "LEGAL_DOMAIN_RULE_INJECTED" in prompt
        assert "[도메인 스킬]" in prompt

    @pytest.mark.asyncio
    async def test_skill_비활성이면_주입_안함(self, tmp_path, monkeypatch):
        (tmp_path / "legal.yaml").write_text(
            "name: legal\n"
            "triggers: [조항]\n"
            "prompt_addon: LEGAL_RULE_INJECTED\n",
            encoding="utf-8")

        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_plan,
            _FakeToolResult,
            _collect)
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.router import QueryRouter

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])

        config_ns = SimpleNamespace(
            rag=SimpleNamespace(
                agent=SimpleNamespace(
                    enabled=True,
                    max_iterations=3,
                    stream_reasoning=False,
                    planner_enabled=True,
                    verifier_enabled=True,
                    verifier_max_repairs=1,
                    legacy_fallback_enabled=True,
                    session_source_reuse=False,
                    session_source_reuse_limit=5,
                    parallel_steps=False,
                    clarifier_enabled=False,
                    clarifier_max_questions=2,
                    personas_enabled=False,
                    native_thinking=False,
                    skills_enabled=False,  # 비활성
                    skills_dir=str(tmp_path)),
                request_timeout=60.0))

        async def _simple_fn(q):
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=QueryRouter(agent_enabled=True),
            app_state=fixtures.app_state,
            config=config_ns,
            ollama_model="test",
            api_base="http://localhost:11434",
            rag_max_tokens=-1,
            simple_stream_fn=_simple_fn)
        await _collect(orch.handle_agent("제15조 조항"))

        prompt = fixtures.app_state.http_client.stream.call_args.kwargs["json"]["prompt"]
        assert "LEGAL_RULE_INJECTED" not in prompt
