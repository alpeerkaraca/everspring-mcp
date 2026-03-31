"""EverSpring MCP - ChromaDB client wrapper."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .config import VectorConfig


class ChromaClient:
    """Wrapper around ChromaDB persistent client."""
    
    def __init__(self, config: VectorConfig) -> None:
        self.config = config
        self._client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self._collection: Collection | None = None
    
    def get_collection(self) -> Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
            )
        return self._collection
    
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        collection = self.get_collection()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    
    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection = self.get_collection()
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )


__all__ = ["ChromaClient"]
