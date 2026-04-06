"""EverSpring MCP - Embedding generation utilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.logging import get_logger

logger = get_logger("vector.embeddings")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding result for a chunk."""
    
    chunk_id: str
    vector: list[float]


class Embedder:
    """Embedding generator for markdown chunks.
    
    Uses sentence-transformers for embedding generation.
    """
    
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        start = time.perf_counter()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        logger.info(f"Embedder initialized (model={model_name}) in {time.perf_counter() - start:.3f}s")
    
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            import_start = time.perf_counter()
            logger.info("Importing sentence_transformers...")
            from sentence_transformers import SentenceTransformer
            logger.info(
                f"sentence_transformers imported in {time.perf_counter() - import_start:.2f}s"
            )

            start = time.perf_counter()
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded in {time.perf_counter() - start:.2f}s")
        return self._model

    def ensure_model_loaded(self) -> SentenceTransformer:
        """Ensure embedding model is loaded and return it."""
        return self._get_model()

    async def prefetch_model(self) -> None:
        """Preload embedding model so artifacts are cached before first query."""
        await asyncio.to_thread(self.ensure_model_loaded)
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts asynchronously."""
        model = self.ensure_model_loaded()
        embeddings = await asyncio.to_thread(model.encode, texts, convert_to_numpy=True)
        # Convert numpy arrays to lists of floats
        return [emb.tolist() for emb in embeddings]
    
    async def embed_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = await self.embed_texts(batch)
            vectors.extend(batch_vectors)
        return vectors


__all__ = ["EmbeddingResult", "Embedder"]
