"""EverSpring MCP - Hybrid retrieval with RRF fusion."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from ..models.metadata import SearchResult
from ..utils.logging import get_logger
from .bm25_index import BM25Index
from .chroma_client import ChromaClient
from .config import VectorConfig
from .embeddings import Embedder

logger = get_logger("vector.retriever")


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from:
    - Dense retrieval: ChromaDB cosine similarity
    - Sparse retrieval: BM25 keyword matching
    
    RRF formula: score(d) = Σ 1/(k + rank(d)) for each ranker
    """
    
    RRF_K: int = 60  # RRF smoothing constant
    
    def __init__(
        self,
        config: VectorConfig | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        """Initialize hybrid retriever.
        
        Args:
            config: Vector configuration
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
        """
        start = time.perf_counter()
        self.config = config or VectorConfig.from_env()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self._chroma = ChromaClient(self.config)
        self._bm25 = BM25Index(self.config)
        self._embedder = Embedder(
            model_name=self.config.embedding_model,
            batch_size=self.config.batch_size,
        )
        logger.info(f"HybridRetriever initialized in {time.perf_counter() - start:.2f}s")
    
    async def search(
        self,
        query: str,
        top_k: int = 3,
        module: str | None = None,
        version_major: int | None = None,
        fetch_k: int = 20,
        deduplicate_urls: bool = True,
    ) -> list[SearchResult]:
        """Search using hybrid retrieval with RRF fusion.
        
        Args:
            query: Search query text
            top_k: Number of final results to return
            module: Optional module filter
            version_major: Optional major version filter
            fetch_k: Number of candidates to fetch from each retriever
            deduplicate_urls: If True, return only best chunk per URL
            
        Returns:
            List of SearchResult sorted by RRF score (descending)
        """
        # Build filter for ChromaDB
        where = self._build_where_filter(module, version_major)
        
        # Run both retrievals in parallel
        logger.debug(f"Searching query='{query[:50]}...' top_k={top_k} module={module} version={version_major}")
        dense_task = self._dense_search(query, fetch_k, where)
        sparse_task = asyncio.to_thread(
            self._bm25.search,
            query,
            fetch_k,
            module,
            version_major,
        )
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )
        logger.debug(f"Dense: {len(dense_results)} results, Sparse: {len(sparse_results)} results")
        
        # Extract ranked ID lists
        dense_ranking = [r["id"] for r in dense_results]
        sparse_ranking = [r.doc_id for r in sparse_results]
        
        # Compute RRF scores
        rrf_scores = self._reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking],
            [self.dense_weight, self.sparse_weight],
        )
        
        # Build content/metadata lookup from dense results
        content_lookup: dict[str, str] = {}
        metadata_lookup: dict[str, dict[str, Any]] = {}
        
        for r in dense_results:
            content_lookup[r["id"]] = r["content"]
            metadata_lookup[r["id"]] = r["metadata"]
        
        # For docs only in sparse results, fetch from BM25 index
        for r in sparse_results:
            if r.doc_id not in content_lookup:
                doc_data = self._bm25.get_document(r.doc_id)
                if doc_data:
                    content, meta = doc_data
                    content_lookup[r.doc_id] = content
                    metadata_lookup[r.doc_id] = meta
        
        # Build rank lookups
        dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ranking, 1)}
        sparse_ranks = {doc_id: rank for rank, doc_id in enumerate(sparse_ranking, 1)}
        
        # Sort by RRF score and take top results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        
        for doc_id, score in sorted_docs:
            if doc_id not in content_lookup:
                continue
            
            meta = metadata_lookup[doc_id]
            url = str(meta.get("url", "")).strip()

            # Skip duplicate URLs if deduplication enabled
            if deduplicate_urls and url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            
            results.append(SearchResult(
                id=doc_id,
                content=content_lookup[doc_id],
                title=meta.get("title", ""),
                url=url,
                module=meta.get("module", ""),
                submodule=meta.get("submodule") or None,
                version_major=meta.get("version_major", 0),
                version_minor=meta.get("version_minor", 0),
                score=score,
                dense_rank=dense_ranks.get(doc_id),
                sparse_rank=sparse_ranks.get(doc_id),
                section_path=meta.get("section_path", ""),
                has_code=meta.get("has_code", False),
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    async def _dense_search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Run dense retrieval via ChromaDB.
        
        Returns list of dicts with id, content, metadata, distance.
        """
        # Embed query
        query_vectors = await self._embedder.embed_texts([query])
        
        # Query ChromaDB
        results = self._chroma.query(
            query_embeddings=query_vectors,
            n_results=top_k,
            where=where,
        )
        
        # Flatten results
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        
        return docs
    
    def _reciprocal_rank_fusion(
        self,
        rankings: list[list[str]],
        weights: list[float],
    ) -> dict[str, float]:
        """Compute RRF scores from multiple rankings.
        
        Args:
            rankings: List of ranked ID lists (best first)
            weights: Weight for each ranking
            
        Returns:
            Dict mapping doc_id to fused RRF score
        """
        scores: dict[str, float] = defaultdict(float)
        
        for ranking, weight in zip(rankings, weights):
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] += weight * (1.0 / (self.RRF_K + rank))
        
        return dict(scores)
    
    @staticmethod
    def _build_where_filter(
        module: str | None,
        version_major: int | None,
    ) -> dict[str, Any] | None:
        """Build ChromaDB where filter from parameters."""
        conditions = []
        
        if module:
            conditions.append({"module": {"$eq": module}})
        if version_major is not None:
            conditions.append({"version_major": {"$eq": version_major}})
        
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
    
    def build_bm25_index(self) -> None:
        """Build BM25 index from ChromaDB collection.
        
        Call this after indexing new documents.
        """
        collection = self._chroma.get_collection()
        
        # Fetch all documents from ChromaDB
        all_docs = collection.get(
            include=["documents", "metadatas"],
        )
        
        if not all_docs["ids"]:
            return
        
        self._bm25.build(
            doc_ids=all_docs["ids"],
            documents=all_docs["documents"] or [],
            metadatas=all_docs["metadatas"] or [],
        )
        self._bm25.save()
    
    def ensure_bm25_index(self) -> bool:
        """Ensure BM25 index is loaded.
        
        Returns:
            True if index is ready, False if needs building
        """
        if self._bm25.is_loaded:
            return True
        return self._bm25.load()


__all__ = ["HybridRetriever", "SearchResult"]
