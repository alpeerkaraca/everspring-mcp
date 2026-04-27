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
        logger.debug(f"Upserting {len(ids)} vectors")
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
        """Compatibility wrapper for the Search API (not available in local).
        Attempts to translate search objects back to legacy query calls if possible,
        but primarily intended to be bypassed by the retriever using query() directly.
        """
        # For now, we will revert the retriever to call query() or search() as a wrapper
        # if the search_obj has query_embeddings, n_results, etc.
        # But a cleaner approach is for retriever to use .query() directly.
        # Here we just implement a barebones wrapper or raise if not easily translatable.
        logger.warning("Search API called on local Chroma client. Falling back to query API.")
        return self.query(
            query_embeddings=search_obj.get("query_embeddings", []),
            n_results=search_obj.get("n_results", 5),
            where=search_obj.get("where"),
        )

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
