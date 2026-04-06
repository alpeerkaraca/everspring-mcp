"""EverSpring MCP - Spring Documentation Search Tool.

Implements the vector search tool with score thresholds,
progress notifications, and module/version filtering.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable

from ..models.metadata import SearchResult
from ..utils.logging import get_logger
from ..vector.chroma_client import ChromaClient
from ..vector.config import VectorConfig
from ..vector.retriever import HybridRetriever
from .models import (
    ModuleInfo,
    ProgressNotification,
    SearchParameters,
    SearchResponse,
    SearchResultItem,
    SearchStatus,
    StatusResponse,
)

logger = get_logger("mcp.tools")


class SpringDocsTool:
    """Spring documentation search tool with hybrid retrieval.
    
    Provides:
    - Vector search with cosine + BM25 + RRF fusion
    - Score threshold enforcement
    - Module/version/submodule filtering
    - Progress notifications
    """
    
    # Default score threshold - results below this are rejected
    DEFAULT_SCORE_THRESHOLD: float = 0.01
    
    # Progress stages
    STAGE_INIT = "initializing"
    STAGE_EMBEDDING = "embedding_query"
    STAGE_DENSE = "dense_search"
    STAGE_SPARSE = "sparse_search"
    STAGE_FUSION = "rrf_fusion"
    STAGE_FILTERING = "applying_filters"
    STAGE_COMPLETE = "complete"
    
    def __init__(
        self,
        config: VectorConfig | None = None,
        progress_callback: Callable[[ProgressNotification], None] | None = None,
    ) -> None:
        """Initialize the search tool.
        
        Args:
            config: Vector configuration
            progress_callback: Optional callback for progress notifications
        """
        start = time.perf_counter()
        self.config = config or VectorConfig.from_env()
        self._progress_callback = progress_callback
        self._retriever: HybridRetriever | None = None
        self._chroma: ChromaClient | None = None
        self._initialized = False
        logger.info(f"SpringDocsTool initialized in {time.perf_counter() - start:.3f}s")
    
    def _notify_progress(
        self,
        stage: str,
        message: str,
        percentage: float,
        **details: Any,
    ) -> None:
        """Send progress notification if callback is registered."""
        if self._progress_callback:
            notification = ProgressNotification(
                stage=stage,
                message=message,
                percentage=percentage,
                details=details,
            )
            self._progress_callback(notification)
        
        # Also log progress
        logger.info(f"[{percentage:.0f}%] {stage}: {message}")
    
    async def initialize(self) -> bool:
        """Initialize retriever and indexes.
        
        Returns:
            True if initialization successful
        """
        start = time.perf_counter()
        self._notify_progress(
            self.STAGE_INIT,
            "Loading vector database and indexes...",
            0.0,
        )
        
        try:
            self._chroma = ChromaClient(self.config)
            self._retriever = HybridRetriever(self.config)
            
            self._notify_progress(
                self.STAGE_INIT,
                "Loading BM25 sparse index...",
                30.0,
            )
            
            # Ensure BM25 index is loaded
            bm25_loaded = self._retriever.ensure_bm25_index()
            if not bm25_loaded:
                logger.warning("BM25 index not found; sparse search disabled")
                self._notify_progress(
                    self.STAGE_INIT,
                    "BM25 index not loaded - using dense search only",
                    50.0,
                    bm25_available=False,
                )
            else:
                self._notify_progress(
                    self.STAGE_INIT,
                    "BM25 index loaded",
                    50.0,
                    bm25_available=True,
                )
            
            self._initialized = True
            self._notify_progress(
                self.STAGE_INIT,
                "Initialization complete",
                100.0,
            )
            logger.info(f"Full tool initialization in {time.perf_counter() - start:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search tool: {e}")
            self._notify_progress(
                self.STAGE_INIT,
                f"Initialization failed: {e}",
                0.0,
                error=str(e),
            )
            return False
    
    async def search(
        self,
        params: SearchParameters,
    ) -> SearchResponse:
        """Search Spring documentation.
        
        Args:
            params: Search parameters
            
        Returns:
            SearchResponse with results or error status
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        if not self._retriever:
            return SearchResponse(
                status=SearchStatus.ERROR,
                message="Search tool not initialized",
                query=params.query,
                results_found=0,
                results_returned=0,
                score_threshold=params.score_threshold,
            )
        
        self._notify_progress(
            self.STAGE_EMBEDDING,
            f"Processing query: '{params.query[:50]}...'",
            10.0,
            query_length=len(params.query),
        )
        
        try:
            # Build filters info for response
            filters_applied: dict[str, Any] = {}
            if params.module:
                filters_applied["module"] = params.module
            if params.version:
                filters_applied["version"] = params.version
            if params.submodule:
                filters_applied["submodule"] = params.submodule
            
            self._notify_progress(
                self.STAGE_DENSE,
                "Running hybrid search (dense + sparse)...",
                30.0,
                filters=filters_applied,
            )
            
            # Run hybrid search
            # Fetch more candidates than needed for threshold filtering
            fetch_k = params.top_k * 5
            raw_results = await self._retriever.search(
                query=params.query,
                top_k=fetch_k,
                module=params.module,
                version_major=params.version,
                fetch_k=fetch_k,
                deduplicate_urls=True,
            )
            
            self._notify_progress(
                self.STAGE_FUSION,
                f"Found {len(raw_results)} candidates",
                60.0,
                candidates=len(raw_results),
            )
            
            # Apply score threshold filtering
            self._notify_progress(
                self.STAGE_FILTERING,
                f"Applying score threshold ({params.score_threshold})...",
                70.0,
            )
            
            warnings: list[str] = []
            filtered_results: list[SearchResult] = []
            
            for result in raw_results:
                # Apply submodule filter if specified
                if params.submodule:
                    result_submodule = result.submodule or ""
                    if result_submodule.lower() != params.submodule.lower():
                        continue
                
                # Apply score threshold
                if result.score >= params.score_threshold:
                    filtered_results.append(result)
            
            # Limit to requested top_k
            filtered_results = filtered_results[:params.top_k]
            
            self._notify_progress(
                self.STAGE_FILTERING,
                f"{len(filtered_results)} results above threshold",
                85.0,
                above_threshold=len(filtered_results),
            )
            
            # Build response
            if not filtered_results:
                if raw_results:
                    # Had results but all below threshold
                    max_score = max(r.score for r in raw_results) if raw_results else 0
                    status = SearchStatus.BELOW_THRESHOLD
                    message = (
                        f"Found {len(raw_results)} results but all scores "
                        f"({max_score:.4f} max) are below threshold "
                        f"({params.score_threshold}). Try broadening your query."
                    )
                    warnings.append(
                        f"Best score {max_score:.4f} is below threshold {params.score_threshold}"
                    )
                else:
                    status = SearchStatus.NO_RESULTS
                    message = "No results found. Try different search terms or remove filters."
            else:
                status = SearchStatus.SUCCESS
                message = f"Found {len(filtered_results)} relevant results"
            
            # Convert to response items
            result_items = [
                self._to_result_item(idx + 1, result)
                for idx, result in enumerate(filtered_results)
            ]
            
            self._notify_progress(
                self.STAGE_COMPLETE,
                message,
                100.0,
                status=status.value,
                results=len(result_items),
            )
            
            return SearchResponse(
                status=status,
                message=message,
                query=params.query,
                filters_applied=filters_applied,
                results_found=len(raw_results),
                results_returned=len(result_items),
                score_threshold=params.score_threshold,
                results=result_items,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return SearchResponse(
                status=SearchStatus.ERROR,
                message=f"Search failed: {e}",
                query=params.query,
                results_found=0,
                results_returned=0,
                score_threshold=params.score_threshold,
            )
    
    def _to_result_item(self, rank: int, result: SearchResult) -> SearchResultItem:
        """Convert SearchResult to SearchResultItem."""
        # Build score breakdown string
        breakdown_parts = []
        if result.dense_rank is not None:
            breakdown_parts.append(f"dense: #{result.dense_rank}")
        if result.sparse_rank is not None:
            breakdown_parts.append(f"sparse: #{result.sparse_rank}")
        breakdown = ", ".join(breakdown_parts) if breakdown_parts else "N/A"
        
        # Format version string
        version = f"v{result.version_major}.{result.version_minor}"
        
        return SearchResultItem(
            rank=rank,
            title=result.title,
            url=result.url,
            module=result.module,
            version=version,
            submodule=result.submodule,
            score=result.score,
            score_breakdown=breakdown,
            content=result.content,
            has_code=result.has_code,
        )
    
    async def get_status(self) -> StatusResponse:
        """Get server/index status.
        
        Returns:
            StatusResponse with index health info
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._chroma:
            return StatusResponse(
                healthy=False,
                index_ready=False,
                total_documents=0,
                bm25_index_loaded=False,
            )
        
        try:
            collection = self._chroma.get_collection()
            total_docs = collection.count()
            
            # Get unique modules and versions
            all_docs = collection.get(include=["metadatas"])
            module_data: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"versions": set(), "submodules": set(), "count": 0}
            )
            
            if all_docs["metadatas"]:
                for meta in all_docs["metadatas"]:
                    mod = meta.get("module", "unknown")
                    module_data[mod]["versions"].add(meta.get("version_major", 0))
                    if meta.get("submodule"):
                        module_data[mod]["submodules"].add(meta.get("submodule"))
                    module_data[mod]["count"] += 1
            
            modules = [
                ModuleInfo(
                    name=name,
                    versions=sorted(data["versions"]),
                    submodules=sorted(data["submodules"]),
                    doc_count=data["count"],
                )
                for name, data in sorted(module_data.items())
            ]
            
            bm25_loaded = self._retriever.ensure_bm25_index() if self._retriever else False
            
            return StatusResponse(
                healthy=True,
                index_ready=total_docs > 0,
                modules=modules,
                total_documents=total_docs,
                bm25_index_loaded=bm25_loaded,
            )
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return StatusResponse(
                healthy=False,
                index_ready=False,
                total_documents=0,
                bm25_index_loaded=False,
            )
    
    async def list_available_modules(self) -> list[str]:
        """Get list of available modules.
        
        Returns:
            List of module names
        """
        status = await self.get_status()
        return [m.name for m in status.modules]


__all__ = ["SpringDocsTool"]
