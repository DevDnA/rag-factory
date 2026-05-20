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
