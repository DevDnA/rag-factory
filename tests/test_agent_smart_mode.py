"""Phase 15a — smart_mode 프리셋 테스트.

``smart_mode=True``이면 P0 Phase 기능 전체가 자동 활성화되는지 검증.
"""

from __future__ import annotations

import pytest

from slm_factory.config import AgentRagConfig


class TestSmartModeCascade:
    def test_기본값은_모두_False(self):
        cfg = AgentRagConfig()
        assert cfg.smart_mode is False
        assert cfg.intent_classifier_enabled is False
        assert cfg.clarifier_enabled is False
        assert cfg.personas_enabled is False
        assert cfg.review_work_enabled is False
        assert cfg.planner_enabled is False
        assert cfg.reflector_enabled is False

    def test_smart_mode_True이면_P0_전체_활성(self):
        cfg = AgentRagConfig(smart_mode=True)
        assert cfg.intent_classifier_enabled is True
        assert cfg.clarifier_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.review_work_enabled is True
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.reflector_enabled is True
        assert cfg.legacy_fallback_enabled is True

    def test_smart_mode_False이고_개별_True는_존중(self):
        cfg = AgentRagConfig(
            smart_mode=False,
            planner_enabled=True,
            reflector_enabled=True,
        )
        assert cfg.planner_enabled is True
        assert cfg.reflector_enabled is True
        assert cfg.clarifier_enabled is False
        assert cfg.personas_enabled is False

    def test_smart_mode_True는_추가_플래그도_켜짐(self):
        """smart_mode=True + 개별 False → smart_mode가 승리(OR 합산)."""
        cfg = AgentRagConfig(
            smart_mode=True,
            # 개별 False 명시했지만 smart_mode가 True이면 override
            clarifier_enabled=False,
        )
        # smart_mode가 cascade하므로 결과는 True
        assert cfg.clarifier_enabled is True

    def test_P1_P2_플래그는_영향_없음(self):
        """smart_mode는 P0만 활성화 — P1(parallel_steps, session_source_reuse 등) 영향 없음."""
        cfg = AgentRagConfig(smart_mode=True)
        # P1/P2 플래그는 기본값 유지
        # parallel_steps는 기본 False (위험성 있어 opt-in)
        assert cfg.parallel_steps is False
        assert cfg.persist_sessions is False


class TestUltraModeCascade:
    """Phase 15b — ultra_mode는 smart_mode + P1/P2 전체."""

    def test_ultra_mode_True이면_P0_P1_P2_전부_활성(self):
        cfg = AgentRagConfig(ultra_mode=True)
        # P0
        assert cfg.intent_classifier_enabled is True
        assert cfg.clarifier_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.review_work_enabled is True
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.reflector_enabled is True
        assert cfg.legacy_fallback_enabled is True
        # P1/P2
        assert cfg.hooks_enabled is True
        assert cfg.memory_compression_enabled is True
        assert cfg.self_improvement_enabled is True
        assert cfg.review_work_retry is True
        assert cfg.session_source_reuse is True

    def test_ultra_mode는_파일기반_기능은_건드리지_않음(self):
        """skills_enabled, custom_personas_dir는 디렉터리 지정 필수이므로 자동 활성화 안 함."""
        cfg = AgentRagConfig(ultra_mode=True)
        assert cfg.skills_enabled is False
        assert cfg.custom_personas_dir == ""
        assert cfg.persist_sessions is False
        assert cfg.parallel_steps is False

    def test_ultra_mode_False에서_개별_True_존중(self):
        cfg = AgentRagConfig(
            ultra_mode=False,
            smart_mode=False,
            hooks_enabled=True,
            self_improvement_enabled=True,
        )
        assert cfg.hooks_enabled is True
        assert cfg.self_improvement_enabled is True
        # P0는 비활성 유지
        assert cfg.planner_enabled is False


class TestQualityModeCascade:
    """고품질 답변 프리셋 — Ralph 통합 quality loop을 진입점으로 사용."""

    def test_기본값은_quality_mode_False(self):
        cfg = AgentRagConfig()
        assert cfg.quality_mode is False
        assert cfg.ralph_loop_enabled is False
        assert cfg.intent_verbalization_enabled is False

    def test_quality_mode_True이면_핵심_플래그_자동_활성(self):
        cfg = AgentRagConfig(quality_mode=True)
        assert cfg.ralph_loop_enabled is True
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.intent_classifier_enabled is True
        assert cfg.intent_verbalization_enabled is True
        assert cfg.clarifier_enabled is True
        assert cfg.personas_enabled is True
        assert cfg.session_source_reuse is True
        assert cfg.legacy_fallback_enabled is True

    def test_quality_mode_True여도_ralph_loop_enabled_명시_False면_OFF(self):
        # latency 우선 모드 — quality_mode의 다른 기능은 살리되 ralph만 끔.
        cfg = AgentRagConfig(quality_mode=True, ralph_loop_enabled=False)
        assert cfg.ralph_loop_enabled is False
        # 다른 cascade 필드는 그대로 유지.
        assert cfg.planner_enabled is True
        assert cfg.verifier_enabled is True
        assert cfg.intent_classifier_enabled is True
        assert cfg.clarifier_enabled is True

    def test_quality_mode_True이면_max_iterations_1도_존중(self):
        # latency 우선 시 max_iter=1이 유효 (retry 없이 단일 평가).
        # 0 이하만 권장값(3)으로 끌어올림.
        cfg = AgentRagConfig(quality_mode=True, ralph_loop_max_iterations=1)
        assert cfg.ralph_loop_max_iterations == 1

    def test_quality_mode_True이면_max_iterations_5_명시는_존중(self):
        cfg = AgentRagConfig(quality_mode=True, ralph_loop_max_iterations=5)
        assert cfg.ralph_loop_max_iterations == 5

    def test_quality_mode_smart_mode_동시_활성도_안전(self):
        cfg = AgentRagConfig(quality_mode=True, smart_mode=True)
        # smart_mode의 직렬 chain 플래그가 켜져 있어도 ralph_loop_enabled가
        # orchestrator에서 우회 게이트 역할을 하므로 답변 경로는 ralph 단일.
        assert cfg.ralph_loop_enabled is True
        assert cfg.review_work_enabled is True  # smart_mode가 true로 설정
        assert cfg.reflector_enabled is True

    def test_quality_mode_False면_개별_True_존중(self):
        cfg = AgentRagConfig(
            quality_mode=False,
            ralph_loop_enabled=True,
            intent_verbalization_enabled=True,
        )
        assert cfg.ralph_loop_enabled is True
        assert cfg.intent_verbalization_enabled is True
        # planner는 자동 활성화되지 않음.
        assert cfg.planner_enabled is False

    def test_quality_mode는_threshold_strategy_등_세부값은_건드리지_않음(self):
        cfg = AgentRagConfig(
            quality_mode=True,
            ralph_loop_quality_threshold=8.5,
            ralph_loop_strategy="reset",
        )
        assert cfg.ralph_loop_quality_threshold == 8.5
        assert cfg.ralph_loop_strategy == "reset"
