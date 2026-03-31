"""EverSpring MCP - Embedding generation utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable

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
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
    
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts asynchronously."""
        model = self._get_model()
        return await asyncio.to_thread(model.encode, texts, convert_to_numpy=False)
    
    async def embed_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = await self.embed_texts(batch)
            vectors.extend([list(v) for v in batch_vectors])
        return vectors


__all__ = ["EmbeddingResult", "Embedder"]
