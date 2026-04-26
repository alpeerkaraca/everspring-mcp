"""EverSpring MCP - ChromaDB client wrapper."""

from __future__ import annotations

import time
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from everspring_mcp.utils.logging import get_logger
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import MAIN_TIER

logger = get_logger("vector.chroma")


class ChromaClient:
    """Wrapper around ChromaDB persistent client."""

    def __init__(self, config: VectorConfig) -> None:
        start = time.perf_counter()
        self.config = config
        self._client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self._collection: Collection | None = None
        logger.info(f"ChromaClient initialized in {time.perf_counter() - start:.2f}s")

    def get_collection(self) -> Collection:
        if self._collection is None:
            logger.debug(f"Getting collection: {self.config.collection_name}")
            
            schema = None
            if self.config.embedding_tier == MAIN_TIER:
                from chromadb import Schema, SparseVectorIndexConfig, K
                schema = Schema()
                # We define a sparse index on the "sparse_embedding" key.
                # source_key=K.DOCUMENT means it can be auto-generated from document text,
                # but we will also support manual upserts of sparse vectors.
                schema.create_index(
                    SparseVectorIndexConfig(source_key=K.DOCUMENT),
                    key="sparse_embedding"
                )
                logger.info(f"Using native sparse schema for tier: {self.config.embedding_tier}")

            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                schema=schema
            )
        return self._collection

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]] | dict[str, list[Any]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        collection = self.get_collection()
        # logger.debug(f"Upserting {len(ids)} vectors")
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

    def search(self, search_obj: Any) -> Any:
        """Execute a search using the new Search API (available with schema)."""
        collection = self.get_collection()
        if not hasattr(collection, "search"):
            raise RuntimeError(
                f"Collection '{self.config.collection_name}' does not support search API. "
                "Ensure it was created with a Schema."
            )
        return collection.search(search_obj)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Delete vectors by IDs or metadata filter."""
        collection = self.get_collection()
        collection.delete(ids=ids, where=where)

    def count(self) -> int:
        """Count vectors in current collection."""
        return self.get_collection().count()

    def reset_collection(self) -> None:
        """Delete and recreate the configured collection."""
        self._client.delete_collection(name=self.config.collection_name)
        self._collection = None


__all__ = ["ChromaClient"]
