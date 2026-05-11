"""ParsedDocument 및 QAPair 데이터 모델의 단위 테스트입니다."""

from slm_factory.models import ParsedDocument, QAPair


class TestParsedDocument:
    """ParsedDocument 데이터클래스의 테스트입니다."""

    def test_기본_생성_및_필드_검증(self, make_parsed_doc):
        """ParsedDocument를 기본값으로 생성하고 필드를 검증합니다."""
        doc = make_parsed_doc()
        assert doc.doc_id == "test.pdf"
        assert doc.title == "테스트 문서"
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0

    def test_기본값_빈_리스트_및_빈_딕셔너리(self):
        """tables와 metadata의 기본값이 빈 리스트와 빈 딕셔너리인지 확인합니다."""
        doc = ParsedDocument(doc_id="a.pdf", title="제목", content="내용")
        assert doc.tables == []
        assert doc.metadata == {}

    def test_커스텀_필드_지정(self):
        """모든 필드를 명시적으로 지정하여 올바르게 저장되는지 확인합니다."""
        tables = ["| A | B |"]
        metadata = {"author": "홍길동", "pages": 10}
        doc = ParsedDocument(
            doc_id="report.pdf",
            title="보고서",
            content="본문 내용",
            tables=tables,
            metadata=metadata,
        )
        assert doc.doc_id == "report.pdf"
        assert doc.title == "보고서"
        assert doc.content == "본문 내용"
        assert doc.tables == tables
        assert doc.metadata == metadata

    def test_기본값_독립성(self):
        """서로 다른 인스턴스의 기본 mutable 필드가 공유되지 않는지 확인합니다."""
        doc1 = ParsedDocument(doc_id="a", title="t", content="c")
        doc2 = ParsedDocument(doc_id="b", title="t", content="c")
        doc1.tables.append("table")
        assert doc2.tables == []


class TestQAPair:
    """QAPair 데이터클래스의 테스트입니다."""

    def test_기본_생성_및_필드_검증(self, make_qa_pair):
        """QAPair를 기본값으로 생성하고 필드를 검증합니다."""
        pair = make_qa_pair()
        assert pair.question == "테스트 질문입니다."
        assert len(pair.answer) > 0
        assert pair.source_doc == "test.pdf"
        assert pair.category == "general"

    def test_기본값_빈_문자열(self):
        """instruction, source_doc, category의 기본값이 빈 문자열인지 확인합니다."""
        pair = QAPair(question="질문", answer="답변")
        assert pair.instruction == ""
        assert pair.source_doc == ""
        assert pair.category == ""

    def test_커스텀_필드_지정(self):
        """모든 필드를 명시적으로 지정하여 올바르게 저장되는지 확인합니다."""
        pair = QAPair(
            question="Q",
            answer="A",
            instruction="지시문",
            source_doc="doc.pdf",
            category="factual",
        )
        assert pair.question == "Q"
        assert pair.answer == "A"
        assert pair.instruction == "지시문"
        assert pair.source_doc == "doc.pdf"
        assert pair.category == "factual"
