from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import Embedder

async def _run_model_cache(args: argparse.Namespace) -> int:
    config = VectorConfig.from_env()
    updates: dict[str, Any] = {}
    if args.embed_model:
        updates["embedding_model"] = args.embed_model
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if updates:
        config = config.model_copy(update=updates)

    logger = logging.getLogger(__name__)
    start = asyncio.get_running_loop().time()
    logger.info("Prefetching embedding model cache for %s", config.embedding_model)

    embedder = Embedder(
        model_name=config.embedding_model,
        batch_size=config.batch_size,
        tier=config.embedding_tier,
    )
    await embedder.prefetch_model()

    elapsed_seconds = asyncio.get_running_loop().time() - start
    payload = {
        "status": "success",
        "embedding_model": config.embedding_model,
        "batch_size": config.batch_size,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    logger.info(
        "Embedding model cache ready (model=%s) in %.2fs",
        config.embedding_model,
        elapsed_seconds,
    )

    if args.json:
        logger.info(json.dumps(payload, indent=2))
    else:
        logger.info(
            f"Model cache ready for {payload['embedding_model']} "
            f"in {payload['elapsed_seconds']:.2f}s"
        )
    return 0

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    model_cache = subparsers.add_parser(
        "model-cache",
        help="Warm embedding model cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_cache.add_argument(
        "--embed-model", default=None, help="Embedding model override"
    )
    model_cache.add_argument(
        "--batch-size", type=int, default=None, help="Embedding batch size override"
    )
    model_cache.add_argument("--json", action="store_true", help="Output JSON summary")
    model_cache.set_defaults(func=_run_model_cache)
