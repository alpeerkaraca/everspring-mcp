"""EverSpring MCP - Development server entry point.

Run with: 
  cd C:\\Users\\AlperKaraca\\Desktop\\everspring_mcp
  uv run mcp dev src/everspring_mcp/mcp/dev_server.py:mcp

Uses lazy initialization - heavy components load on first tool call.
First search takes ~12s (model loading), subsequent searches are fast.

Logs are written to: logs/mcp_server_YYYY-MM-DD.log
"""

import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Add src to path for development
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Use centralized logging (logs to logs/ directory)
from everspring_mcp.utils.logging import setup_logging, get_logger

setup_logging(level="INFO", console=True, file=True, name="mcp_server")
logger = get_logger("mcp.dev_server")

# Create FastMCP instance directly (instant - no heavy imports)
mcp = FastMCP("everspring-mcp")
logger.info("FastMCP instance created")

# Lazy-loaded components (None until first use)
_retriever = None
_initialized = False


async def _ensure_initialized():
    """Lazy load heavy components on first use."""
    global _retriever, _initialized
    
    if _initialized:
        return _retriever
    
    start = time.perf_counter()
    logger.info("First tool call - initializing (this takes ~12s)...")
    
    # Import heavy modules only when needed
    import_start = time.perf_counter()
    from everspring_mcp.vector.config import VectorConfig
    from everspring_mcp.vector.retriever import HybridRetriever
    logger.info(
        f"Imported vector modules in {time.perf_counter() - import_start:.2f}s"
    )
    
    config_start = time.perf_counter()
    config = VectorConfig.from_env()
    logger.info(f"Loaded VectorConfig in {time.perf_counter() - config_start:.2f}s")

    retriever_start = time.perf_counter()
    _retriever = HybridRetriever(config)
    logger.info(f"Built HybridRetriever in {time.perf_counter() - retriever_start:.2f}s")
    
    # Load BM25 index
    bm25_start = time.perf_counter()
    _retriever.ensure_bm25_index()
    logger.info(f"BM25 ready in {time.perf_counter() - bm25_start:.2f}s")
    
    # Warm up embedding model with a dummy query
    logger.info("Warming up embedding model...")
    warmup_start = time.perf_counter()
    await _retriever.search("warmup", top_k=1)
    logger.info(f"Warmup query finished in {time.perf_counter() - warmup_start:.2f}s")
    
    _initialized = True
    logger.info(f"Initialization complete in {time.perf_counter() - start:.2f}s")
    
    return _retriever


@mcp.tool()
async def search_spring_docs(
    query: str,
    module: str | None = None,
    version: int | None = None,
    submodule: str | None = None,
    top_k: int = 3,
    score_threshold: float = 0.01,
) -> dict:
    """Search Spring documentation using hybrid retrieval.
    
    Uses cosine similarity + BM25 with Reciprocal Rank Fusion to find
    the most relevant documentation chunks.
    
    NOTE: First search takes ~12s to load the embedding model.
    Subsequent searches are fast (<1s).
    
    Args:
        query: Natural language query (e.g., "how to configure DataSource")
        module: Filter by Spring module (e.g., "spring-boot", "spring-framework")
        version: Filter by major version (e.g., 4 for Spring Boot 4.x)
        submodule: Filter by submodule (e.g., "redis" for spring-data-redis)
        top_k: Number of results to return (1-10, default 3)
        score_threshold: Minimum relevance score (0.0-1.0, default 0.01)
        
    Returns:
        Search results with status, scores, and documentation content.
    """
    retriever = await _ensure_initialized()
    
    # Run search
    results = await retriever.search(
        query=query,
        top_k=top_k * 3,  # Fetch more for filtering
        module=module,
        version_major=version,
        deduplicate_urls=True,
    )
    
    # Apply score threshold and submodule filter
    filtered = []
    for r in results:
        if r.score < score_threshold:
            continue
        if submodule and (r.submodule or "").lower() != submodule.lower():
            continue
        filtered.append(r)
        if len(filtered) >= top_k:
            break
    
    # Build response
    if not filtered:
        if results:
            max_score = max(r.score for r in results)
            return {
                "status": "below_threshold",
                "message": f"Found {len(results)} results but best score ({max_score:.4f}) is below threshold ({score_threshold})",
                "query": query,
                "results": [],
            }
        return {
            "status": "no_results",
            "message": "No results found",
            "query": query,
            "results": [],
        }
    
    return {
        "status": "success",
        "message": f"Found {len(filtered)} results",
        "query": query,
        "filters": {"module": module, "version": version, "submodule": submodule},
        "results": [
            {
                "rank": i + 1,
                "title": r.title,
                "url": r.url,
                "module": r.module,
                "version": f"v{r.version_major}.{r.version_minor}",
                "submodule": r.submodule,
                "score": round(r.score, 4),
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
                "has_code": r.has_code,
                "content": r.content,
            }
            for i, r in enumerate(filtered)
        ],
    }


@mcp.tool()
async def get_spring_docs_status() -> dict:
    """Get EverSpring MCP server status.
    
    Returns information about index health, available modules, and document counts.
    NOTE: First call takes ~12s to initialize.
    """
    retriever = await _ensure_initialized()
    
    # Get collection stats
    collection = retriever._chroma.get_collection()
    total_docs = collection.count()
    
    # Get module breakdown
    all_docs = collection.get(include=["metadatas"])
    modules = {}
    
    if all_docs["metadatas"]:
        for meta in all_docs["metadatas"]:
            mod = meta.get("module", "unknown")
            if mod not in modules:
                modules[mod] = {"versions": set(), "submodules": set(), "count": 0}
            modules[mod]["versions"].add(meta.get("version_major", 0))
            if meta.get("submodule"):
                modules[mod]["submodules"].add(meta.get("submodule"))
            modules[mod]["count"] += 1
    
    return {
        "healthy": True,
        "index_ready": total_docs > 0,
        "total_documents": total_docs,
        "bm25_loaded": retriever._bm25.is_loaded,
        "modules": [
            {
                "name": name,
                "versions": sorted(data["versions"]),
                "submodules": sorted(data["submodules"]),
                "doc_count": data["count"],
            }
            for name, data in sorted(modules.items())
        ],
    }


@mcp.tool()
async def list_spring_modules() -> list[str]:
    """List available Spring modules for filtering searches."""
    status = await get_spring_docs_status()
    return [m["name"] for m in status["modules"]]



