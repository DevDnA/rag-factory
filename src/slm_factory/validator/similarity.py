"""QA 쌍에 대한 임베딩 기반 근거성(Groundedness) 검증.

필수 요구사항: sentence-transformers: uv sync --extra validation
답변 임베딩과 원본 문서 청크 간의 코사인 유사도를 사용하여
답변이 원본 자료에 근거하고 있는지 검증합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import GroundednessConfig
    from ..models import QAPair

from ..utils import get_logger

logger = get_logger("validator.similarity")


def _check_sentence_transformers() -> None:
    """sentence-transformers가 없으면 설치 지침과 함께 ImportError를 발생시킵니다."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "Groundedness validation requires sentence-transformers. "
            "Install with: uv sync --extra validation"
        )


class GroundednessChecker:
    """임베딩 유사도를 사용하여 QA 답변이 원본 문서에 근거하고 있는지 확인합니다.

    sentence-transformers를 사용하여 답변 텍스트와 원본 문서 청크를 임베딩한 후
    코사인 유사도를 계산합니다. 임계값 이하의 답변은 잠재적 환각(Hallucination)으로 표시됩니다.
    """

    def __init__(self, config: GroundednessConfig) -> None:
        _check_sentence_transformers()
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.threshold = config.threshold
        self._model = SentenceTransformer(config.model)
        logger.info("Loaded embedding model: %s", config.model)

    def _chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 64
    ) -> list[str]:
        """임베딩 비교를 위해 텍스트를 격치는 청크로 분할합니다.

        매개변수:
            text: 원본 문서 텍스트.
            chunk_size: 청크당 문자 수.
            overlap: 청크 간 문자 격치.

        반환값:
            텍스트 청크 리스트.
        """
        if not text:
            return []
        # overlap이 chunk_size 이상이면 무한루프 방지
        if overlap >= chunk_size:
            overlap = chunk_size // 4
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks if chunks else [text]

    def score(self, answer: str, source_text: str) -> float:
        """근거성 점수를 계산합니다: 답변과 원본 청크 간의 최대 코사인 유사도.

        매개변수:
            answer: 생성된 답변 텍스트.
            source_text: 전체 원본 문서 텍스트.

        반환값:
            [0, 1] 범위의 실수 — 높을수록 더 근거가 있음.
        """
        from sentence_transformers import util as st_util

        chunks = self._chunk_text(source_text)

        answer_embedding = self._model.encode(
            answer, convert_to_tensor=True, show_progress_bar=False
        )
        chunk_embeddings = self._model.encode(
            chunks, convert_to_tensor=True, show_progress_bar=False
        )

        similarities = st_util.cos_sim(answer_embedding, chunk_embeddings)
        max_sim = float(similarities.max())
        return max_sim

    def check(self, pair: QAPair, source_text: str) -> tuple[bool, float]:
        """QA 쌍의 답변이 원본 텍스트에 근거하고 있는지 확인합니다.

        매개변수:
            pair: 확인할 QA 쌍.
            source_text: 전체 원본 문서 텍스트.

        반환값:
            (근거_여부, 유사도_점수) 튜플.
        """
        sim = self.score(pair.answer, source_text)
        grounded = sim >= self.threshold
        if not grounded:
            logger.warning(
                f"Low groundedness ({sim:.3f} < {self.threshold}): "
                f"Q: {pair.question[:80]}..."
            )
        return grounded, sim

    def check_batch(
        self,
        pairs: list[QAPair],
        source_texts: dict[str, str],
    ) -> tuple[list[QAPair], list[tuple[QAPair, float]]]:
        """QA 쌍 배치에 대한 근거성을 확인합니다.

        매개변수:
            pairs: QA 쌍 리스트.
            source_texts: source_doc ID를 문서 텍스트에 매핑하는 딕셔너리.

        반환값:
            (근거있는_쌍, 근거없는_쌍_및_점수) 튜플.
        """
        from sentence_transformers import util as st_util

        grounded = []
        ungrounded = []
        chunk_cache: dict[str, object] = {}  # source_doc → chunk_embeddings 텐서

        for pair in pairs:
            source = source_texts.get(pair.source_doc, "")
            if not source:
                logger.warning(
                    "No source text for doc '%s', skipping groundedness check",
                    pair.source_doc,
                )
                grounded.append(pair)
                continue

            # 청크 임베딩 캐시 — 동일 문서의 청크는 한 번만 인코딩
            if pair.source_doc not in chunk_cache:
                chunks = self._chunk_text(source)
                chunk_cache[pair.source_doc] = self._model.encode(
                    chunks, convert_to_tensor=True, show_progress_bar=False
                )

            answer_emb = self._model.encode(
                pair.answer, convert_to_tensor=True, show_progress_bar=False
            )
            sim = float(st_util.cos_sim(answer_emb, chunk_cache[pair.source_doc]).max())

            if sim >= self.threshold:
                grounded.append(pair)
            else:
                ungrounded.append((pair, sim))
                logger.warning(
                    f"Low groundedness ({sim:.3f} < {self.threshold}): "
                    f"Q: {pair.question[:80]}..."
                )

        logger.info(
            f"Groundedness check: {len(grounded)}/{len(pairs)} grounded, "
            f"{len(ungrounded)}/{len(pairs)} below threshold ({self.threshold})"
        )
        return grounded, ungrounded
