from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from chromadb.errors import InvalidArgumentError

from everspring_mcp.mcp.prompt import PromptBuilder
from everspring_mcp.storage.repository import StorageManager
from everspring_mcp.vector.chroma_client import ChromaClient
from everspring_mcp.vector.config import VectorConfig, chunk_defaults_for_tier
from everspring_mcp.vector.embeddings import default_model_for_tier
from everspring_mcp.vector.indexer import VectorIndexer
from everspring_mcp.vector.retriever import HybridRetriever
from everspring_mcp.cli.utils import (
    resolve_vector_config,
    _render_search_results,
    console,
)

logger = logging.getLogger("everspring_mcp")


async def _run_index(args: argparse.Namespace) -> int:
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module for index --reindex")

    logger.debug("Resolving vector config...")
    config = await resolve_vector_config(args)
    updates: dict[str, Any] = {}
    default_chunk_size, default_overlap = chunk_defaults_for_tier(config.embedding_tier)
    if args.max_tokens is not None:
        updates["max_tokens"] = args.max_tokens
    else:
        updates["max_tokens"] = default_chunk_size
    if args.overlap_tokens is not None:
        updates["overlap_tokens"] = args.overlap_tokens
    else:
        updates["overlap_tokens"] = default_overlap
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.chunk_workers is not None:
        updates["chunk_workers"] = args.chunk_workers
    if args.upsert_batch_size is not None:
        updates["chroma_upsert_batch_size"] = args.upsert_batch_size
    if hasattr(args, "exclude") and args.exclude is not None:
        updates["exclude_patterns"] = args.exclude
    if updates:
        config = config.model_copy(update=updates)

    reset_count = 0
    deleted_vectors = 0
    logger.info(f"Starting indexing with config: {config.model_dump_json(indent=2)}")
    if args.reindex:
        logger.debug(f"Initializing StorageManager for reindex: db_path={config.db_path}")
        storage = StorageManager(config.db_path)
        logger.debug("Connecting to storage...")
        await storage.connect()
        try:
            module_filter = args.module
            major_filter = args.version
            submodule_filter = args.submodule if module_filter else None
            logger.debug(f"Resetting indexed docs: module={module_filter}, major={major_filter}, submodule={submodule_filter}")
            reset_count = await storage.documents.reset_indexed(
                module=module_filter,
                major=major_filter,
                submodule=submodule_filter,
            )
        finally:
            logger.debug("Closing storage connection...")
            await storage.close()

        logger.debug(f"Initializing ChromaClient for reindex: tier={config.embedding_tier}")
        chroma = ChromaClient(config)
        where_conditions: list[dict[str, Any]] = []
        if args.module:
            where_conditions.append({"module": {"$eq": args.module}})
        if args.version is not None:
            where_conditions.append({"version_major": {"$eq": args.version}})
        if args.submodule and args.module:
            where_conditions.append({"submodule": {"$eq": args.submodule}})
        if args.submodule is None and args.module:
            where_conditions.append({"submodule": {"$eq": ""}})

        where_filter: dict[str, Any] | None
        if not where_conditions:
            where_filter = None
        elif len(where_conditions) == 1:
            where_filter = where_conditions[0]
        else:
            where_filter = {"$and": where_conditions}

        if where_filter is None:
            logger.debug("Resetting full Chroma collection...")
            before_delete = chroma.count()
            chroma.reset_collection()
            deleted_vectors = before_delete
        else:
            logger.debug(f"Deleting from Chroma with filter: {where_filter}")
            before_delete = chroma.count()
            chroma.delete(where=where_filter)
            after_delete = chroma.count()
            deleted_vectors = max(0, before_delete - after_delete)

    logger.debug("Initializing VectorIndexer...")
    async with VectorIndexer(config=config) as indexer:
        if getattr(args, "all", False):
            logger.debug("Calculating total unindexed documents...")
            # Calculate total number of documents in DB to override limit
            limit = await indexer._storage.documents.count()
        else:
            limit = args.limit
        logger.debug(f"Starting indexing loop: limit={limit}")
        stats = await indexer.index_unindexed(limit=limit)

    bm25_index_built = False
    if args.build_bm25 and config.embedding_tier != "main":
        logger.debug(f"Building BM25 index for tier={config.embedding_tier}")
        retriever = HybridRetriever(config=config)
        retriever.build_bm25_index()
        bm25_index_built = True
    elif args.build_bm25:
        logger.info(
            "Skipping BM25 build for tier=main; using model-native retrieval path"
        )

    payload = {
        "documents_indexed": stats.documents_indexed,
        "chunks_indexed": stats.chunks_indexed,
        "documents_reset": reset_count,
        "vectors_deleted": deleted_vectors,
        "bm25_index_built": bm25_index_built,
        "bm25_index_path": (
            str(config.data_dir / "bm25_index.pkl") if bm25_index_built else None
        ),
    }
    if args.json:
        logger.info(json.dumps(payload, indent=2))
    else:
        message = (
            f"Indexed {payload['documents_indexed']} documents "
            f"({payload['chunks_indexed']} chunks)"
        )
        if bm25_index_built:
            message = f"{message}; BM25 index built"
        logger.info(message)
    return 0


async def _run_search(args: argparse.Namespace) -> int:
    logger.debug("Resolving vector config...")
    config = await resolve_vector_config(args)
    logger.debug(f"Initializing HybridRetriever: tier={config.embedding_tier}")
    retriever = HybridRetriever(config=config)

    if config.embedding_tier != "main" and (
        args.build_index or not retriever.ensure_bm25_index()
    ):
        logger.debug("BM25 index missing or rebuild requested. Building...")
        retriever.build_bm25_index()

    try:
        logger.debug(
            f"Executing hybrid search: query='{args.query}', top_k={args.top_k}, "
            f"module={args.module}, version={args.version}"
        )
        results = await retriever.search(
            query=args.query,
            top_k=args.top_k,
            module=args.module,
            version_major=args.version,
            deduplicate_urls=not args.no_dedup,
        )
    except InvalidArgumentError as exc:
        error_text = str(exc)
        if "dimension" in error_text.lower():
            raise SystemExit(
                "Embedding dimension mismatch detected between the selected tier/model and "
                "the local Chroma snapshot. Stop other processes using "
                f"'{config.chroma_dir}', then run:\n"
                f"  python -m everspring_mcp.main sync --mode snapshot-download "
                f'--snapshot-model "{config.embedding_model}" --snapshot-tier {config.embedding_tier}\n'
                "After sync completes, rerun the search command."
            ) from exc
        raise

    if getattr(args, "prompt", False):
        logger.debug("Building LLM prompt context...")
        builder = (
            PromptBuilder()
            .add_user_query(args.query)
            .add_filters(module=args.module, version=args.version)
            .add_retrieved_context(results)
        )
        console.print(builder.build())
        return 0

    if args.json:
        output = [
            {
                "id": r.id,
                "title": r.title,
                "url": r.url,
                "module": r.module,
                "version": f"{r.version_major}.{r.version_minor}",
                "score": round(r.score, 4),
                "dense_rank": r.dense_rank,
                "sparse_rank": r.sparse_rank,
                "content": r.content,
            }
            for r in results
        ]
        logger.info(json.dumps(output, indent=2))
    else:
        logger.debug("Rendering search results to console...")
        _render_search_results(results)

    return 0


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    index = subparsers.add_parser(
        "index",
        help="Index docs into ChromaDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index.add_argument("--limit", type=int, default=50, help="Max documents to index")
    index.add_argument(
        "--all",
        action="store_true",
        help="Index all unindexed documents (ignores --limit)",
    )
    index.add_argument("--data-dir", default=None, help="Local data directory override")
    index.add_argument(
        "--db-filename", default=None, help="SQLite database filename override"
    )
    index.add_argument(
        "--docs-subdir", default=None, help="Local docs subdirectory override"
    )
    index.add_argument(
        "--chroma-dir", default=None, help="ChromaDB persistent dir override"
    )
    index.add_argument(
        "--collection", default=None, help="ChromaDB collection name override"
    )
    index.add_argument("--embed-model", default=None, help="Embedding model override")
    index.add_argument(
        "--tier",
        choices=["main", "slim", "xslim"],
        default="main",
        help="Embedding tier",
    )
    index.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens per chunk override"
    )
    index.add_argument(
        "--overlap-tokens", type=int, default=None, help="Token overlap override"
    )
    index.add_argument(
        "--batch-size", type=int, default=None, help="Embedding batch size override"
    )
    index.add_argument(
        "--chunk-workers",
        type=int,
        default=None,
        help="Parallel workers for document chunk preparation override",
    )
    index.add_argument(
        "--upsert-batch-size",
        type=int,
        default=None,
        help="Chroma upsert batch size override",
    )
    index.add_argument(
        "--reindex", action="store_true", help="Reset indexed flags and rebuild vectors"
    )
    index.add_argument(
        "--build-bm25",
        action="store_true",
        help="Build BM25 index after vector indexing",
    )
    index.add_argument(
        "--exclude",
        nargs="*",
        default=[
            "**/index-all.html",
            "**/allclasses-*.html",
            "**/allpackages-index.html",
            "**/*-tree.html",
            "**/deprecated-list.html",
            "**/constant-values.html",
            "**/serialized-form.html",
            "**/help-doc.html",
            "**/overview-summary.html",
            "**/search.js",
            "**/search-index.js",
        ],
        help="Glob patterns to hard-exclude from indexing",
    )
    index.add_argument(
        "--module", default=None, help="Module filter for --reindex (e.g., spring-boot)"
    )
    index.add_argument(
        "--version", type=int, default=None, help="Major version filter for --reindex"
    )
    index.add_argument(
        "--submodule", default=None, help="Submodule filter for --reindex"
    )
    index.add_argument("--json", action="store_true", help="Output JSON summary")
    index.set_defaults(func=_run_index)

    search = subparsers.add_parser(
        "search",
        help="Search indexed docs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    search.add_argument(
        "--query",
        "-q",
        required=True,
        help="Search query text",
    )
    search.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of results to return",
    )
    search.add_argument(
        "--module",
        default=None,
        help="Filter by Spring module",
    )
    search.add_argument(
        "--version",
        type=int,
        default=None,
        help="Filter by major version",
    )
    search.add_argument(
        "--build-index",
        action="store_true",
        help="Rebuild BM25 index before searching",
    )
    search.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable URL deduplication (show multiple chunks from same page)",
    )
    search.add_argument(
        "--tier",
        choices=["main", "slim", "xslim"],
        default="main",
        help="Embedding tier",
    )
    search.add_argument(
        "--prompt",
        action="store_true",
        help="Output final LLM-formatted prompt context (Markdown)",
    )
    search.add_argument("--json", action="store_true", help="Output JSON results")
    search.set_defaults(func=_run_search)
