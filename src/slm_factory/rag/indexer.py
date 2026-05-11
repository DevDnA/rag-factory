"""Qdrant 벡터 인덱싱 — corpus.parquet을 임베딩하여 Qdrant에 적재합니다.

``CorpusExporter``가 생성한 ``corpus.parquet`` 파일을 읽어
sentence-transformers 모델로 임베딩한 뒤 Qdrant에 upsert합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("rag.indexer")


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """None 값을 제거한 메타데이터를 반환합니다.

    Qdrant 페이로드는 dict/list를 네이티브로 지원하므로
    JSON 문자열 변환 없이 그대로 저장합니다.
    """
    return {k: v for k, v in metadata.items() if v is not None}


class RAGIndexer:
    """corpus.parquet을 Qdrant 벡터 DB에 임베딩하여 적재합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.db_path = Path(config.paths.output) / config.rag.vector_db_path

    def index(self, corpus_path: Path) -> Path:
        """corpus.parquet을 읽어 Qdrant에 임베딩 후 upsert합니다.

        매개변수
        ----------
        corpus_path:
            ``corpus.parquet`` 파일 경로.

        반환값
        -------
        Path
            Qdrant 저장 경로.
        """
        import pandas as pd

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers가 설치되지 않았습니다. "
                "uv sync --extra rag 로 설치하세요."
            )

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError:
            raise RuntimeError(
                "qdrant-client가 설치되지 않았습니다. uv sync --extra rag 로 설치하세요."
            )

        df = pd.read_parquet(corpus_path)
        logger.info("corpus.parquet 로드 완료 — %d개 청크", len(df))

        embedding_model = self.config.rag.embedding_model
        logger.info("임베딩 모델 로드: %s", embedding_model)
        model = SentenceTransformer(embedding_model)
        logger.info("임베딩 모델 로드 완료 (device=%s)", model.device.type)

        self.db_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(self.db_path))

        collection_name = self.config.rag.collection_name

        doc_ids = df["doc_id"].tolist()
        contents = df["contents"].tolist()
        metadatas_raw = (
            df["metadata"].tolist() if "metadata" in df.columns else [{}] * len(df)
        )

        batch_size = self.config.rag.batch_size
        total = len(contents)
        collection_created = False

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_contents = contents[start:end]
            batch_ids = doc_ids[start:end]
            batch_metadatas = metadatas_raw[start:end]

            logger.info("임베딩 인코딩 중: %d~%d / %d 청크", start + 1, end, total)
            embeddings = model.encode(batch_contents, show_progress_bar=True)

            if not collection_created:
                dim = embeddings.shape[1]
                if not client.collection_exists(collection_name):
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                collection_created = True
                logger.info(
                    "Qdrant 컬렉션 준비 완료: %s (dim=%d)", collection_name, dim
                )

            points = []
            for i, (vec, doc, doc_id, meta) in enumerate(
                zip(embeddings.tolist(), batch_contents, batch_ids, batch_metadatas)
            ):
                payload: dict[str, Any] = {"document": doc, "doc_id": doc_id}
                if isinstance(meta, dict):
                    sanitized = _sanitize_metadata(meta)
                    if "parent_content" in meta and meta["parent_content"]:
                        sanitized["parent_content"] = str(meta["parent_content"])[
                            :10000
                        ]
                    payload.update(sanitized)
                points.append(PointStruct(id=start + i, vector=vec, payload=payload))

            client.upsert(collection_name=collection_name, points=points)
            logger.info(
                "임베딩 upsert 진행: %d/%d 청크 완료",
                end,
                total,
            )

        client.close()
        logger.info(
            "Qdrant 인덱싱 완료 — %d개 청크, 저장 경로: %s",
            total,
            self.db_path,
        )
        return self.db_path
