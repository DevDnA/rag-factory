"""smart_mode 프리셋 cascade 검증.

``smart_mode=True``이면 Agent RAG의 핵심 컴포넌트(planner, verifier,
intent_classifier, clarifier, personas, legacy_fallback, session_source_reuse)
가 한 번에 활성화되는지 확인합니다.
"""

from __future__ import annotations

from slm_factory.config import AgentRagConfig


class TestSmartModeCascade:
    def test_기본값은_smart_mode_False(self):
        cfg = AgentRagConfig()
        assert cfg.smart_mode is False
        assert cfg.planner_enabled is False
        assert cfg.verifier_enabled is True  # 기본 True (planner 켜질 때만 동작)
        assert cfg.intent_classifier_enabled is False
        assert cfg.clarifier_enabled is False
        assert cfg.personas_enabled is False
        assert cfg.legacy_fallback_enabled is True  # 기본 True
        assert cfg.session_source_reuse is True  # 기본 True

    def test_smart_mode_True이면_핵심_플래그_자동_활성(self):
        cfg = AgentRagConfig(smart_mode=True)
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.legacy_fallback_enabled is True
        assert cfg.intent_classifier_enabled is True
        assert cfg.clarifier_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.session_source_reuse is True

    def test_smart_mode_False이고_개별_True는_존중(self):
        cfg = AgentRagConfig(
            smart_mode=False,
            planner_enabled=True,
            personas_enabled=True,
        )
        assert cfg.planner_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.intent_classifier_enabled is False
        assert cfg.clarifier_enabled is False

    def test_smart_mode_True는_개별_False_override(self):
        """smart_mode=True + 개별 False → smart_mode가 승리(cascade는 강제 ON)."""
        cfg = AgentRagConfig(
            smart_mode=True,
            clarifier_enabled=False,
        )
        # smart_mode가 cascade하므로 결과는 True
        assert cfg.clarifier_enabled is True

    def test_smart_mode는_의도_발화는_건드리지_않음(self):
        """intent_verbalization은 명시 opt-in. smart_mode와 독립."""
        cfg = AgentRagConfig(smart_mode=True)
        assert cfg.intent_verbalization_enabled is False

    def test_smart_mode는_파일기반_기능은_건드리지_않음(self):
        """skills_enabled, custom_personas_dir, parallel_steps 등은 명시 설정 필요."""
        cfg = AgentRagConfig(smart_mode=True)
        assert cfg.skills_enabled is False
        assert cfg.custom_personas_dir == ""
        assert cfg.parallel_steps is False
        assert cfg.persist_sessions is False
