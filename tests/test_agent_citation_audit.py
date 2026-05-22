"""citation_audit 단위 테스트 — 인용 토큰 추출과 환각 인용 검출."""

from __future__ import annotations

from rag_factory.rag.agent.citation_audit import audit_citations, extract_citations


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtract:
    def test_단일_인용(self):
        assert extract_citations("개정 [doc:law_15.pdf].") == ["law_15.pdf"]

    def test_복수_인용(self):
        text = "동의 [doc:a.pdf]. 세부 [doc:b.pdf]."
        assert extract_citations(text) == ["a.pdf", "b.pdf"]

    def test_중복_제거(self):
        text = "[doc:x.pdf] 그리고 다시 [doc:x.pdf]."
        assert extract_citations(text) == ["x.pdf"]

    def test_빈_답변(self):
        assert extract_citations("") == []
        assert extract_citations("출처 없음") == []

    def test_빈_token은_무시(self):
        assert extract_citations("[doc:]") == []

    def test_공백_token은_무시(self):
        assert extract_citations("[doc:  ]") == []

    def test_순서_보존(self):
        text = "[doc:z.pdf] then [doc:a.pdf] then [doc:m.pdf]"
        assert extract_citations(text) == ["z.pdf", "a.pdf", "m.pdf"]


# ---------------------------------------------------------------------------
# audit_citations
# ---------------------------------------------------------------------------


class TestAudit:
    def test_모든_인용_매칭(self):
        sources = [
            {"doc_id": "law_15.pdf", "content": "...", "score": 0.9},
            {"doc_id": "guide.pdf", "content": "...", "score": 0.8},
        ]
        answer = "동의 요건 [doc:law_15.pdf]. 절차는 [doc:guide.pdf]."
        assert audit_citations(answer, sources) == []

    def test_미매칭_인용_검출(self):
        sources = [{"doc_id": "law_15.pdf", "content": "...", "score": 0.9}]
        answer = "동의 [doc:law_15.pdf]. 그리고 [doc:nonexistent.pdf]."
        assert audit_citations(answer, sources) == ["nonexistent.pdf"]

    def test_doc_id의_chunk_suffix_prefix_매칭(self):
        """source doc_id가 'rfp.pdf::p5' 형태여도 답변의 [doc:rfp.pdf]는 매칭."""
        sources = [{"doc_id": "rfp.pdf::p5", "content": "...", "score": 0.9}]
        answer = "내용은 [doc:rfp.pdf]에 있습니다."
        assert audit_citations(answer, sources) == []

    def test_doc_id의_hash_suffix_prefix_매칭(self):
        """source doc_id가 'guide.pdf#chunk_3' 형태여도 매칭."""
        sources = [{"doc_id": "guide.pdf#chunk_3", "content": "...", "score": 0.9}]
        answer = "참고 [doc:guide.pdf]."
        assert audit_citations(answer, sources) == []

    def test_case_insensitive_매칭(self):
        sources = [{"doc_id": "Law_15.PDF", "content": "...", "score": 0.9}]
        answer = "참고 [doc:law_15.pdf]."
        assert audit_citations(answer, sources) == []

    def test_source_list_빈_경우_모든_인용_환각(self):
        answer = "[doc:a.pdf] [doc:b.pdf]"
        assert audit_citations(answer, []) == ["a.pdf", "b.pdf"]

    def test_답변에_인용_없으면_빈_리스트(self):
        sources = [{"doc_id": "law.pdf", "content": "...", "score": 0.9}]
        assert audit_citations("출처 없음", sources) == []

    def test_source의_doc_id가_None이면_무시(self):
        sources = [
            {"doc_id": None, "content": "...", "score": 0.9},
            {"doc_id": "law.pdf", "content": "...", "score": 0.9},
        ]
        answer = "[doc:law.pdf]"
        assert audit_citations(answer, sources) == []

    def test_source가_doc_id_대신_source_키_사용(self):
        """일부 코드 경로에서 source dict가 'source' 키를 쓸 수 있음."""
        sources = [{"source": "fallback.pdf", "content": "...", "score": 0.9}]
        answer = "[doc:fallback.pdf]"
        assert audit_citations(answer, sources) == []

    def test_복합_시나리오(self):
        """매칭 2개 + 환각 1개."""
        sources = [
            {"doc_id": "a.pdf::p1", "content": "...", "score": 0.9},
            {"doc_id": "b.pdf::p2", "content": "...", "score": 0.8},
        ]
        answer = "[doc:a.pdf] / [doc:b.pdf] / [doc:c.pdf]"
        assert audit_citations(answer, sources) == ["c.pdf"]


# ---------------------------------------------------------------------------
# Nested-bracket 파일명 (한국 법령문서 명명 관습)
# ---------------------------------------------------------------------------


class TestNestedBracketFilenames:
    """``[행정] 사건.pdf`` 처럼 brackets prefix가 포함된 파일명 처리."""

    def test_행정_brackets_prefix_파일명_추출(self):
        answer = "[doc:[행정] 사건.pdf]"
        assert extract_citations(answer) == ["[행정] 사건.pdf"]

    def test_특허_brackets_prefix_파일명_추출(self):
        answer = "판단 [doc:[특허] 원고 제품들의 실시.pdf]"
        assert extract_citations(answer) == ["[특허] 원고 제품들의 실시.pdf"]

    def test_민사_brackets_prefix_파일명_추출(self):
        answer = "[doc:[민사] 스마트폰 성능조절기능.pdf]"
        assert extract_citations(answer) == ["[민사] 스마트폰 성능조절기능.pdf"]

    def test_LLM이_outer_brackets를_이중으로_감싼_경우(self):
        """``[[doc:[특허] X.pdf]]`` — 외곽 brackets는 무시, inner만 추출."""
        answer = "[[doc:[특허] X.pdf]]"
        assert extract_citations(answer) == ["[특허] X.pdf"]

    def test_realistic_법령_full_filename(self):
        """실제 일반화 평가셋 실패 케이스 (allganize_law_010)."""
        answer = (
            "퇴직수당 [doc:[행정] 금품 등 약속이 공무원 재직 중에 이루어지고 "
            "수수가 퇴직 후에 이루어진 경우 공무원연금법에 해당하는 지가 "
            "문제된 사건.pdf]."
        )
        cites = extract_citations(answer)
        assert len(cites) == 1
        assert cites[0].startswith("[행정]")
        assert cites[0].endswith(".pdf")

    def test_nested_brackets_audit_매칭(self):
        """추출된 nested-bracket 파일명이 source list와 정확히 매칭되어 환각으로 분류되지 않음."""
        sources = [
            {
                "source_doc_id": (
                    "[행정] 금품 등 약속이 공무원 재직 중에 이루어지고 "
                    "수수가 퇴직 후에 이루어진 경우 공무원연금법에 해당하는 "
                    "지가 문제된 사건.pdf"
                ),
                "doc_id": "uuid-xxx::p3",
                "content": "...",
                "score": 0.9,
            },
        ]
        answer = (
            "퇴직수당 합산 [doc:[행정] 금품 등 약속이 공무원 재직 중에 "
            "이루어지고 수수가 퇴직 후에 이루어진 경우 공무원연금법에 "
            "해당하는 지가 문제된 사건.pdf]."
        )
        # 환각 없음 — 정확 매칭.
        assert audit_citations(answer, sources) == []

    def test_multiple_nested_bracket_citations_동일_doc(self):
        """동일 nested-bracket 파일을 N회 인용해도 1개로 dedup."""
        answer = (
            "주장 1 [doc:[특허] 사건.pdf]. "
            "주장 2 [doc:[특허] 사건.pdf]. "
            "주장 3 [doc:[특허] 사건.pdf]."
        )
        assert extract_citations(answer) == ["[특허] 사건.pdf"]

    def test_확장자_없는_legacy_uuid_doc_id_fallback(self):
        """확장자 없는 토큰도 fallback regex가 추출 (legacy UUID doc_id 호환)."""
        answer = "결과 [doc:abc-def-uuid] 참고."
        assert extract_citations(answer) == ["abc-def-uuid"]

    def test_extension과_legacy_혼재(self):
        """확장자 있는 토큰 + 확장자 없는 fallback 모두 추출, 순서 보존."""
        answer = "참고 [doc:law.pdf] 그리고 [doc:legacy-uuid-001]."
        assert extract_citations(answer) == ["law.pdf", "legacy-uuid-001"]

    def test_hwp_확장자도_지원(self):
        """``.hwp`` / ``.hwpx`` / ``.docx`` 등 한국 문서 확장자."""
        answer = "[doc:[민사] 사건.hwp] [doc:정책.hwpx] [doc:정관.docx]"
        cites = extract_citations(answer)
        assert "[민사] 사건.hwp" in cites
        assert "정책.hwpx" in cites
        assert "정관.docx" in cites

    def test_대소문자_혼합_확장자(self):
        """``.PDF`` / ``.Hwp`` 등 case-insensitive 확장자."""
        answer = "[doc:[행정] 사건.PDF] [doc:[민사] X.Hwp]"
        cites = extract_citations(answer)
        assert "[행정] 사건.PDF" in cites
        assert "[민사] X.Hwp" in cites
