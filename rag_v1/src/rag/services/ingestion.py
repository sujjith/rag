# src/rag/services/ingestion.py
"""Ingestion service."""
from pathlib import Path
from typing import BinaryIO
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.services.base import BaseService
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunker import SentenceChunker
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.exceptions import DocumentProcessingError, EmbeddingError, VectorStoreError
from rag.models import Document


class IngestionService(BaseService):
    """Service for document ingestion."""

    def __init__(
        self,
        loader: DocumentLoader = None,
        chunker: SentenceChunker = None,
        embedder: Embedder = None,
        store: QdrantStore = None,
    ):
        super().__init__()
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or SentenceChunker()
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()

    def health_check(self) -> dict:
        """Check ingestion service health."""
        return {
            "loader": "ok",
            "chunker": "ok",
            "embedder": self._check_embedder(),
            "store": self._check_store(),
        }

    def _check_embedder(self) -> str:
        try:
            self.embedder.embed(["test"])
            return "ok"
        except Exception as e:
            return f"error: {str(e)}"

    def _check_store(self) -> str:
        try:
            self.store.client.get_collections()
            return "ok"
        except Exception as e:
            return f"error: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def ingest_file(
        self,
        file: BinaryIO,
        filename: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> tuple[str, int]:
        """
        Ingest a file into the RAG system.

        Returns:
            Tuple of (document_id, chunks_created)
        """
        self.logger.info("ingesting_file", filename=filename)

        # Save uploaded file temporarily
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = Path(tmp.name)

        try:
            # Load document
            document = self.loader.load(tmp_path)
            document.metadata["original_filename"] = filename
            self.logger.info("document_loaded", doc_id=document.id, chars=len(document.content))

            # Chunk document
            self.chunker.chunk_size = chunk_size
            self.chunker.overlap = chunk_overlap
            chunks = list(self.chunker.chunk(document))
            self.logger.info("document_chunked", doc_id=document.id, chunks=len(chunks))

            if not chunks:
                raise DocumentProcessingError(
                    "No chunks created from document",
                    {"filename": filename}
                )

            # Generate embeddings
            try:
                texts = [c.content for c in chunks]
                embeddings = self.embedder.embed(texts)
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to generate embeddings: {str(e)}",
                    {"filename": filename}
                )

            # Store in vector database
            try:
                self.store.upsert(chunks)
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to store vectors: {str(e)}",
                    {"filename": filename}
                )

            self.logger.info(
                "ingestion_complete",
                doc_id=document.id,
                chunks=len(chunks),
                filename=filename
            )

            return document.id, len(chunks)

        finally:
            # Cleanup temp file
            tmp_path.unlink(missing_ok=True)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from the store."""
        self.logger.info("deleting_document", doc_id=document_id)
        try:
            self.store.delete_by_document_id(document_id)
            return True
        except Exception as e:
            self.logger.error("delete_failed", doc_id=document_id, error=str(e))
            raise VectorStoreError(
                f"Failed to delete document: {str(e)}",
                {"document_id": document_id}
            )