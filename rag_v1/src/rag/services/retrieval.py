# src/rag/services/retrieval.py
"""Retrieval service."""
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.services.base import BaseService
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.retrieval.reranker import Reranker
from rag.llm.groq import GroqLLM
from rag.models import SearchResult, RAGResponse
from rag.exceptions import EmbeddingError, VectorStoreError, LLMError
from rag.config import settings


class RetrievalService(BaseService):
    """Service for retrieval and generation."""

    def __init__(
        self,
        embedder: Embedder = None,
        store: QdrantStore = None,
        reranker: Reranker = None,
        llm: GroqLLM = None,
    ):
        super().__init__()
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()
        self.reranker = reranker
        self.llm = llm or GroqLLM()

    def health_check(self) -> dict:
        """Check retrieval service health."""
        return {
            "embedder": "ok",
            "store": self._check_store(),
            "llm": self._check_llm(),
        }

    def _check_store(self) -> str:
        try:
            self.store.client.get_collections()
            return "ok"
        except Exception as e:
            return f"error: {str(e)}"

    def _check_llm(self) -> str:
        # Just check if API key is configured
        return "ok" if settings.groq_api_key else "error: no API key"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = False,
        filter_document_id: str = None,
    ) -> tuple[RAGResponse, float]:
        """
        Query the RAG system.

        Returns:
            Tuple of (response, processing_time_ms)
        """
        start_time = time.time()
        self.logger.info("query_started", query=query[:50], top_k=top_k)

        # Embed query
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {str(e)}")

        # Search vector store
        try:
            fetch_k = top_k * 2 if use_reranker else top_k
            results = self.store.search(
                query_vector,
                top_k=fetch_k,
                filter_doc_id=filter_document_id
            )
        except Exception as e:
            raise VectorStoreError(f"Search failed: {str(e)}")

        # Rerank if enabled
        if use_reranker and results:
            if self.reranker is None:
                self.reranker = Reranker()
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # Convert to SearchResult
        search_results = [
            SearchResult(chunk=chunk, score=score)
            for chunk, score in results
        ]

        # Generate response
        try:
            response = self.llm.generate(query, search_results)
        except Exception as e:
            raise LLMError(f"LLM generation failed: {str(e)}")

        processing_time = (time.time() - start_time) * 1000
        self.logger.info(
            "query_complete",
            query=query[:50],
            sources=len(search_results),
            time_ms=processing_time
        )

        return response, processing_time