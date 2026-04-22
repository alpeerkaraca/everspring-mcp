"""Profile initialization times for MCP components."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

logger = logging.getLogger("profile_init")


def timer(name: str):
    """Context manager to time a block."""

    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            logger.info("[START] %s...", name)
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            logger.info("[DONE]  %s: %.2fs", name, elapsed)

    return Timer()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.info("=" * 60)
    logger.info("EverSpring MCP Initialization Profiler")
    logger.info("=" * 60)
    logger.info("")

    # 1. Import VectorConfig
    with timer("Import VectorConfig"):
        from everspring_mcp.vector.config import VectorConfig

    # 2. Create config
    with timer("VectorConfig.from_env()"):
        config = VectorConfig.from_env()

    # 3. Import ChromaClient
    with timer("Import ChromaClient"):
        from everspring_mcp.vector.chroma_client import ChromaClient

    # 4. Create ChromaClient
    with timer("ChromaClient(config)"):
        chroma = ChromaClient(config)

    # 5. Get collection
    with timer("chroma.get_collection()"):
        collection = chroma.get_collection()

    # 6. Count documents
    with timer("collection.count()"):
        count = collection.count()
        logger.info("       Documents: %d", count)

    # 7. Import Embedder
    with timer("Import Embedder"):
        from everspring_mcp.vector.embeddings import Embedder

    # 8. Create Embedder (loads model)
    with timer("Embedder(model_name=...)"):
        embedder = Embedder(
            model_name=config.embedding_model,
            batch_size=config.batch_size,
        )

    # 9. Test embedding
    with timer("embedder.embed_texts(['test'])"):
        embeddings = await embedder.embed_texts(["test query"])
        logger.info("       Vector dim: %d", len(embeddings[0]["dense"]))

    # 10. Import BM25Index
    with timer("Import BM25Index"):
        from everspring_mcp.vector.bm25_index import BM25Index

    # 11. Create BM25Index
    with timer("BM25Index(config)"):
        bm25 = BM25Index(config)

    # 12. Load BM25 index
    with timer("bm25.load()"):
        loaded = bm25.load()
        logger.info("       Loaded: %s", loaded)

    # 13. Import HybridRetriever
    with timer("Import HybridRetriever"):
        from everspring_mcp.vector.retriever import HybridRetriever

    # 14. Create HybridRetriever (combines all)
    with timer("HybridRetriever(config)"):
        retriever = HybridRetriever(config)

    # 15. Ensure BM25 loaded
    with timer("retriever.ensure_bm25_index()"):
        retriever.ensure_bm25_index()

    # 16. Run a test search
    with timer("retriever.search('test query')"):
        results = await retriever.search(
            query="how to configure DataSource",
            top_k=3,
        )
        logger.info("       Results: %d", len(results))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Profiling complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
