"""CorpusProfile 모듈 테스트 — dataclass·persistence·LLM 자동 생성."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_factory.rag.corpus_profile import (
    CorpusProfile,
    extract_acronyms,
    extract_korean_nouns,
    generate_corpus_profile,
    load_corpus_profile,
    merge_with_override,
    save_corpus_profile,
)


class TestCorpusProfile:
    def test_빈_profile은_is_empty_True(self):
        assert CorpusProfile().is_empty() is True

    def test_name만_있어도_not_empty(self):
        assert CorpusProfile(name="X").is_empty() is False

    def test_keywords만_있어도_not_empty(self):
        assert CorpusProfile(keywords=["a"]).is_empty() is False

    def test_to_prompt_header_빈_profile은_빈_문자열(self):
        assert CorpusProfile().to_prompt_header() == ""

    def test_to_prompt_header_모든_필드_포함(self):
        p = CorpusProfile(name="N", summary="S", keywords=["k1", "k2"])
        h = p.to_prompt_header()
        assert "[본 corpus 도메인 정보]" in h
        assert "명칭: N" in h
        assert "요약: S" in h
        assert "k1" in h and "k2" in h

    def test_to_prompt_header_keywords_상한_50개(self):
        p = CorpusProfile(name="N", keywords=[f"k{i}" for i in range(60)])
        h = p.to_prompt_header()
        # k50 이상은 잘림 (cap을 30→50으로 확장: 영문 약어 max 20 + 한국어 명사
        # max 20 + LLM 일반 키워드가 모두 prompt header에 노출되도록 함)
        assert "k0" in h
        assert "k49" in h
        assert "k50" not in h


class TestPersistence:
    def test_없는_파일은_빈_profile(self, tmp_path: Path):
        p = load_corpus_profile(tmp_path / "nope.json")
        assert p.is_empty()

    def test_save_load_왕복(self, tmp_path: Path):
        original = CorpusProfile(
            name="N", summary="S", keywords=["a", "b"], generated_at="2026-05-07T00:00:00Z", model="m"
        )
        path = tmp_path / "profile.json"
        save_corpus_profile(original, path)

        loaded = load_corpus_profile(path)
        assert loaded == original

    def test_파싱_실패_파일은_빈_profile(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not-json", encoding="utf-8")
        p = load_corpus_profile(path)
        assert p.is_empty()

    def test_save_실패는_raise_안함(self):
        # 존재할 수 없는 경로 — 호출만으로는 예외 안 남.
        p = CorpusProfile(name="X")
        save_corpus_profile(p, Path("/proc/1/nope.json"))


class TestMergeOverride:
    def test_override_없으면_auto_그대로(self):
        auto = CorpusProfile(name="auto-N", summary="auto-S", keywords=["k"])
        merged = merge_with_override(auto)
        assert merged.name == "auto-N"
        assert merged.summary == "auto-S"
        assert merged.keywords == ["k"]

    def test_name_override_우선(self):
        auto = CorpusProfile(name="auto-N", summary="auto-S")
        merged = merge_with_override(auto, name_override="user-N")
        assert merged.name == "user-N"
        assert merged.summary == "auto-S"

    def test_keywords_override_완전_대체(self):
        auto = CorpusProfile(keywords=["a", "b", "c"])
        merged = merge_with_override(auto, keywords_override=["x", "y"])
        assert merged.keywords == ["x", "y"]

    def test_빈_keywords_override는_무시(self):
        auto = CorpusProfile(keywords=["a", "b"])
        merged = merge_with_override(auto, keywords_override=[])
        assert merged.keywords == ["a", "b"]


# ---------------------------------------------------------------------------
# generate_corpus_profile — LLM 호출 모킹
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, payload: dict, *, raise_exc: Exception | None = None):
        self._payload = payload
        self._raise = raise_exc
        self.calls = 0

    async def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return _FakeResponse(self._payload)


class TestGenerate:
    @pytest.mark.asyncio
    async def test_정상_생성(self):
        payload = {
            "response": json.dumps({
                "name": "한국 통신사 RFP",
                "summary": "통신 인프라 요구사항.",
                "keywords": ["NMS", "BIS", "MIMO"],
            })
        }
        http = _FakeHttp(payload)
        # 휴리스틱(영문/한국어 명사)이 chunks에서 잡지 않도록 영문 placeholder 사용
        p = await generate_corpus_profile(
            chunks=["sample document one.", "sample document two."],
            http_client=http,
            ollama_model="qwen3.5:9b",
            api_base="http://x",
        )
        assert p.name == "한국 통신사 RFP"
        assert p.summary == "통신 인프라 요구사항."
        # LLM keywords가 그대로 보존되어야 함 (chunks에 휴리스틱 매칭 없음)
        assert "NMS" in p.keywords
        assert "BIS" in p.keywords
        assert "MIMO" in p.keywords
        assert p.model == "qwen3.5:9b"
        assert p.chunks_sampled == 2
        assert p.generated_at  # ISO 시각 채워짐

    @pytest.mark.asyncio
    async def test_빈_청크는_빈_profile(self):
        http = _FakeHttp({"response": ""})
        p = await generate_corpus_profile(
            chunks=[],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.is_empty()
        assert http.calls == 0  # LLM 호출도 안 함

    @pytest.mark.asyncio
    async def test_LLM_예외는_빈_profile(self):
        http = _FakeHttp({}, raise_exc=RuntimeError("boom"))
        p = await generate_corpus_profile(
            chunks=["x"],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.is_empty()

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패는_빈_profile(self):
        payload = {"response": "그냥 텍스트, JSON 아님"}
        http = _FakeHttp(payload)
        p = await generate_corpus_profile(
            chunks=["x"],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.is_empty()

    @pytest.mark.asyncio
    async def test_think_태그_제거(self):
        # qwen 등 thinking 모드 모델의 응답에 포함될 수 있는 태그는 사전 strip.
        payload = {
            "response": (
                "<think>도메인 분석</think>"
                "{\"name\": \"X\", \"summary\": \"Y\", \"keywords\": [\"a\"]}"
            )
        }
        http = _FakeHttp(payload)
        p = await generate_corpus_profile(
            chunks=["x"],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.name == "X"
        # LLM keywords + 휴리스틱 약어 합집합. "a"는 LLM, 이 청크엔 약어 없음.
        assert "a" in p.keywords


class TestExtractAcronyms:
    """extract_acronyms 휴리스틱 추출 동작 검증."""

    def test_telecom_acronyms_detected(self) -> None:
        chunks = [
            "5G 백홀 기반 WiFi 7 AP 대개체. NMS에서 원격 제어. IEEE 802.1x 인증.",
            "5G 주파수는 3.5GHz Sub-6 대역. WPA 3 Enterprise 보안 적용.",
            "AP는 3개 이상 제조사 구성원 필요. NMS로 원격 모니터링. WiFi 7 AP 검증.",
        ]
        result = extract_acronyms(chunks, min_frequency=2)
        assert "5G" in result
        assert "AP" in result
        assert "NMS" in result
        assert "WiFi 7" in result

    def test_camelcase_acronyms_detected(self) -> None:
        # QoS, IoT 같은 camelCase 약어는 [A-Z][a-z]{1,3}[A-Z][a-z]? 패턴이 잡음
        chunks = [
            "QoS 정책에 따른 IoT 디바이스 보안. QoS 우선순위 적용.",
            "IoT 게이트웨이 설치. QoS 측정 및 관리.",
        ]
        result = extract_acronyms(chunks, min_frequency=2)
        assert "QoS" in result
        assert "IoT" in result

    def test_min_frequency_filters_one_off(self) -> None:
        # XYZ는 1회만 등장 — min_frequency=2면 제외
        chunks = ["MIMO XYZ 안테나", "MIMO 안테나 사용", "MIMO 4×4"]
        result = extract_acronyms(chunks, min_frequency=2)
        assert "MIMO" in result
        assert "XYZ" not in result

    def test_stoplist_filters_general_acronyms(self) -> None:
        chunks = ["TV USB PDF 파일", "TV USB PDF 파일", "TV USB PDF 파일"]
        # 모두 stop-list에 포함 → 빈도 충족해도 제외
        assert extract_acronyms(chunks, min_frequency=2) == []

    def test_frequency_descending_order(self) -> None:
        # XYZ/MNO/QRS는 stop-list 미포함 임의 약어
        chunks = [
            "XYZ XYZ XYZ MNO MNO QRS",
            "XYZ MNO",
        ]
        result = extract_acronyms(chunks, min_frequency=2)
        # XYZ(4회), MNO(3회), QRS(1회 → 제외)
        assert result.index("XYZ") < result.index("MNO")
        assert "QRS" not in result

    def test_empty_input(self) -> None:
        assert extract_acronyms([]) == []
        assert extract_acronyms(["", "", ""]) == []

    def test_max_results_caps_output(self) -> None:
        # stop-list/등급 미포함 5종을 모두 빈도 충족 → cap=3 적용 확인
        chunks = [
            "XYZ MNO QRS PQR STU",
            "XYZ MNO QRS PQR STU",
            "XYZ MNO QRS PQR STU",
        ]
        result = extract_acronyms(chunks, min_frequency=2, max_results=3)
        assert len(result) == 3


class TestExtractKoreanNouns:
    """extract_korean_nouns 한국어 명사 추출 검증 — kiwipiepy 의존."""

    def _has_kiwi(self) -> bool:
        try:
            import kiwipiepy  # noqa: F401
            return True
        except ImportError:
            return False

    def test_domain_nouns_detected(self) -> None:
        if not self._has_kiwi():
            pytest.skip("kiwipiepy 미설치")
        chunks = [
            "차등점수제 적용 시 평가점수를 산출한다",
            "수행실적 평가기준에 따른 정량평가",
            "차등점수제와 수행실적, 정량평가 모두 충족",
            "차등점수제 적용은 협상에 의한 계약체결기준에 따라 적용한다",
        ]
        result = extract_korean_nouns(chunks, min_frequency=2)
        # NNG가 잡힌 단어들 (kiwi가 형태소 단위로 쪼개므로 일부 분리됨)
        joined = " ".join(result)
        assert "수행" in joined or "실적" in joined
        assert "평가" in joined or "기준" in joined

    def test_stoplist_excludes_general_nouns(self) -> None:
        if not self._has_kiwi():
            pytest.skip("kiwipiepy 미설치")
        chunks = [
            "사업 내용은 사업 운영을 통해 사업 결과를 제공한다",
            "사업 운영 관리는 사업의 결과로서 결과를 보고한다",
            "사업 정보 자료를 사업 결과로 제공",
        ]
        result = extract_korean_nouns(chunks, min_frequency=2)
        # 모든 단어가 stop-list에 있음 → 빈 리스트
        for blocked in ("사업", "내용", "운영", "결과", "정보", "자료", "제공", "관리"):
            assert blocked not in result

    def test_kiwipiepy_missing_returns_empty(self, monkeypatch) -> None:
        # kiwipiepy import를 가로채서 ImportError 발생시킴
        import sys
        monkeypatch.setitem(sys.modules, "kiwipiepy", None)
        result = extract_korean_nouns(["테스트 문서"], min_frequency=1)
        assert result == []

    def test_empty_input(self) -> None:
        assert extract_korean_nouns([]) == []
        assert extract_korean_nouns(["", ""]) == []
