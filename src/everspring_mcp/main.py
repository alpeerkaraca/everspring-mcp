"""EverSpring MCP CLI entrypoint.

Provides commands for:
- Scraping Spring docs into S3
- Syncing knowledge packs or manifests from S3 into local stores
- Indexing local docs into ChromaDB
- Placeholder server command (not implemented yet)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.models.content import ContentType
from everspring_mcp.scraper.pipeline import PipelineConfig, ScraperPipeline
from everspring_mcp.scraper.registry import SubmoduleRegistry
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.orchestrator import SyncOrchestrator
from everspring_mcp.sync.s3_sync import S3SyncService
from everspring_mcp.storage.repository import StorageManager
from everspring_mcp.vector.chroma_client import ChromaClient
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import Embedder
from everspring_mcp.vector.indexer import VectorIndexer
from everspring_mcp.vector.retriever import HybridRetriever


def _parse_module(value: str) -> SpringModule:
    """Parse module string to SpringModule enum."""
    normalized = value.strip().lower()
    for module in SpringModule:
        if module.value == normalized:
            return module
    raise argparse.ArgumentTypeError(f"Invalid module: {value}")


def _parse_content_type(value: str) -> ContentType:
    """Parse content type string to ContentType enum."""
    normalized = value.strip().lower()
    for content_type in ContentType:
        if content_type.value == normalized:
            return content_type
    raise argparse.ArgumentTypeError(f"Invalid content type: {value}")


def _parse_version(value: str, module: SpringModule) -> SpringVersion:
    """Parse version string to SpringVersion."""
    try:
        return SpringVersion.parse(module, value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EverSpring MCP CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (disabled by default)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser(
        "scrape",
        help="Discover and scrape Spring docs into S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scrape.add_argument("--entry-url", default=None, help="Entry URL for discovery")
    scrape.add_argument("--module", default=None, type=_parse_module, help="Spring module")
    scrape.add_argument("--version", default=None, help="Version string (e.g., 4.0.5)")
    scrape.add_argument("--submodule", default=None, help="Optional submodule key (e.g., redis)")
    scrape.add_argument(
        "--content-type",
        default=ContentType.REFERENCE.value,
        type=_parse_content_type,
        help="Documentation content type",
    )
    scrape.add_argument(
        "--registry-path",
        default=None,
        help="Path to submodule registry JSON (defaults to config/submodules.json)",
    )
    scrape.add_argument("--concurrency", type=int, default=5, help="Max concurrent scrapes")
    scrape.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    scrape.add_argument("--s3-region", default=None, help="S3 region override")
    scrape.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    scrape.add_argument("--no-hash-check", action="store_true", help="Disable hash checks")
    scrape.add_argument("--json", action="store_true", help="Output JSON summary")

    sync = subparsers.add_parser(
        "sync",
        help="Sync raw docs or upload DB snapshots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sync.add_argument("--module", required=False, help="Spring module (e.g., spring-boot)")
    sync.add_argument("--version", required=False, help="Version string (e.g., 4.0.5)")
    sync.add_argument("--submodule", default=None, help="Optional submodule key (e.g., redis)")
    sync.add_argument(
        "--all",
        action="store_true",
        help="Sync all discovered module/submodule/version targets from config\\module_submodule_urls.csv",
    )
    sync.add_argument(
        "--mode",
        choices=["manifest", "snapshot-upload"],
        default="manifest",
        help="Sync mode (manifest downloads raw docs; snapshot-upload backs up local SQLite+Chroma)",
    )
    sync.add_argument(
        "--force",
        action="store_true",
        help="Force sync even if manifest unchanged (manifest mode only)",
    )
    sync.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    sync.add_argument("--s3-region", default=None, help="S3 region override")
    sync.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    sync.add_argument("--data-dir", default=None, help="Local data directory override")
    sync.add_argument("--json", action="store_true", help="Output JSON summary")

    status = subparsers.add_parser(
        "status",
        help="Show local sync status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    status.add_argument("--module", required=False, help="Spring module (e.g., spring-boot)")
    status.add_argument("--version", required=False, help="Version string (e.g., 4.0.5)")
    status.add_argument("--submodule", default=None, help="Optional submodule key (e.g., redis)")
    status.add_argument(
        "--all",
        action="store_true",
        help="Show status for all module/submodule/version entries in local manifest cache",
    )
    status.add_argument("--data-dir", default=None, help="Local data directory override")
    status.add_argument("--json", action="store_true", help="Output JSON summary")

    index = subparsers.add_parser(
        "index",
        help="Index local docs into ChromaDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index.add_argument("--limit", type=int, default=50, help="Max documents to index")
    index.add_argument("--data-dir", default=None, help="Local data directory override")
    index.add_argument("--db-filename", default=None, help="SQLite database filename override")
    index.add_argument("--docs-subdir", default=None, help="Local docs subdirectory override")
    index.add_argument("--chroma-dir", default=None, help="ChromaDB persistent dir override")
    index.add_argument("--collection", default=None, help="ChromaDB collection name override")
    index.add_argument("--embed-model", default=None, help="Embedding model override")
    index.add_argument("--max-tokens", type=int, default=None, help="Max tokens per chunk override")
    index.add_argument("--overlap-tokens", type=int, default=None, help="Token overlap override")
    index.add_argument("--batch-size", type=int, default=None, help="Embedding batch size override")
    index.add_argument("--reindex", action="store_true", help="Reset indexed flags and rebuild vectors")
    index.add_argument("--module", default=None, help="Module filter for --reindex (e.g., spring-boot)")
    index.add_argument("--version", type=int, default=None, help="Major version filter for --reindex")
    index.add_argument("--submodule", default=None, help="Submodule filter for --reindex")
    index.add_argument("--json", action="store_true", help="Output JSON summary")

    search = subparsers.add_parser(
        "search",
        help="Search indexed documentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    search.add_argument(
        "--query", "-q",
        required=True,
        help="Search query text",
    )
    search.add_argument(
        "--top-k", "-k",
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
    search.add_argument("--json", action="store_true", help="Output JSON results")

    model_cache = subparsers.add_parser(
        "model-cache",
        help="Pre-download and cache embedding model artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_cache.add_argument("--embed-model", default=None, help="Embedding model override")
    model_cache.add_argument("--batch-size", type=int, default=None, help="Embedding batch size override")
    model_cache.add_argument("--json", action="store_true", help="Output JSON summary")

    serve = subparsers.add_parser(
        "serve",
        help="Run MCP server (stdio transport)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    serve.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio"],
        help="MCP transport type",
    )
    serve.add_argument("--json", action="store_true", help="Output JSON status on startup")

    # Interactive client command
    client_cmd = subparsers.add_parser(
        "client",
        help="Interactive search client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    client_cmd.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )

    return parser


def _configure_logging(level: str, log_file: str | None) -> None:
    """Configure logging with file output to logs/ directory."""
    from everspring_mcp.utils.logging import setup_logging
    
    # Use centralized logging - always logs to logs/ directory
    setup_logging(
        level=level,
        console=True,
        file=True,  # Always log to file
        name="everspring",
    )
    
    # If user specified additional log file, add it
    if log_file:
        logger = logging.getLogger("everspring_mcp")
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        ))
        logger.addHandler(handler)


async def _run_scrape(args: argparse.Namespace) -> int:
    # Avoid requiring env vars when overrides are provided
    if args.s3_bucket or args.s3_region or args.s3_prefix:
        config = PipelineConfig(
            s3_bucket=args.s3_bucket or "everspring-mcp-kb",
            aws_region=args.s3_region or "eu-central-1",
            s3_prefix=args.s3_prefix or "spring-docs/raw-data",
            enable_hash_check=not args.no_hash_check,
        )
    else:
        config = PipelineConfig.from_env()
    if args.no_hash_check:
        config = config.model_copy(update={"enable_hash_check": False})

    module = args.module
    version = _parse_version(args.version, module) if args.version and module else None
    content_type = args.content_type

    if any([args.entry_url, module, version]) and not all([args.entry_url, module, version]):
        raise ValueError("entry-url, module, and version must be provided together")
    if args.submodule and not module:
        raise ValueError("submodule requires module to be set")

    pipeline = ScraperPipeline(config)
    if args.registry_path:
        registry = SubmoduleRegistry.load(Path(args.registry_path))
        discovery_result, scrape_results = await pipeline.discover_and_scrape_registry(
            registry=registry,
            content_type=content_type,
            concurrency=args.concurrency,
        )
    else:
        discovery_result, scrape_results = await pipeline.discover_and_scrape(
            entry_url=args.entry_url,
            module=module,
            version=version,
            content_type=content_type,
            concurrency=args.concurrency,
            submodule=args.submodule,
        )

    success = sum(1 for r in scrape_results if r.status.value == "success")
    skipped = sum(1 for r in scrape_results if r.status.value == "skipped")
    failed = sum(1 for r in scrape_results if r.status.value == "failed")

    summary = {
        "discovery": {
            "links": discovery_result.link_count,
            "duplicates": discovery_result.duplicates_removed,
            "filtered": discovery_result.filtered_out,
        },
        "results": {
            "success": success,
            "skipped": skipped,
            "failed": failed,
        },
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Discovery: {summary['discovery']['links']} links "
            f"(duplicates: {summary['discovery']['duplicates']}, "
            f"filtered: {summary['discovery']['filtered']})"
        )
        print(
            f"Scrape results: {summary['results']['success']} success, "
            f"{summary['results']['skipped']} skipped, {summary['results']['failed']} failed"
        )
    return 0 if failed == 0 else 1


def _load_sync_targets_from_matrix(matrix_path: Path) -> list[tuple[str, str, str | None]]:
    """Load deduplicated sync targets from module/submodule CSV."""
    if not matrix_path.exists():
        raise SystemExit(f"CSV not found: {matrix_path}")

    import csv

    targets: list[tuple[str, str, str | None]] = []
    seen: set[tuple[str, str, str | None]] = set()
    with matrix_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            module = (row.get("module") or "").strip()
            version = (row.get("version") or "").strip()
            submodule = (row.get("submodule") or "").strip() or None
            if not module or not version:
                continue
            key = (module, version, submodule)
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
    return targets


async def _run_manifest_sync(args: argparse.Namespace, config: SyncConfig) -> int:
    """Run legacy manifest-based incremental sync."""
    async with SyncOrchestrator(config) as orchestrator:
        if args.all:
            targets = _load_sync_targets_from_matrix(Path("config") / "module_submodule_urls.csv")
            results = []
            for module, version, submodule in targets:
                res = await orchestrator.sync_module(
                    module=module,
                    version=version,
                    submodule=submodule,
                    force=args.force,
                )
                results.append(res)

            summary = {
                "mode": "manifest",
                "targets": len(results),
                "completed": sum(1 for r in results if r.status.value == "completed"),
                "failed": sum(1 for r in results if r.status.value == "failed"),
                "files_added": sum(r.files_added for r in results),
                "files_modified": sum(r.files_modified for r in results),
                "files_removed": sum(r.files_removed for r in results),
                "bytes_downloaded": sum(r.bytes_downloaded for r in results),
                "errors": [f"{r.module}:{r.version} -> {err}" for r in results for err in r.errors],
            }
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                print(
                    f"Sync-all (manifest): {summary['completed']}/{summary['targets']} completed, "
                    f"{summary['failed']} failed "
                    f"(+{summary['files_added']} ~{summary['files_modified']} -{summary['files_removed']}, "
                    f"{summary['bytes_downloaded']} bytes)"
                )
                if summary["errors"]:
                    print("Errors:")
                    for err in summary["errors"]:
                        print(f"- {err}")
            return 0 if summary["failed"] == 0 else 1

        result = await orchestrator.sync_module(
            module=args.module,
            version=args.version,
            submodule=args.submodule,
            force=args.force,
        )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(
            f"Sync status (manifest): {result.status.value} "
            f"(+{result.files_added} ~{result.files_modified} -{result.files_removed}, "
            f"{result.bytes_downloaded} bytes)"
        )
        if result.errors:
            print("Errors:")
            for err in result.errors:
                print(f"- {err}")
    return 0 if result.status.value == "completed" else 1


async def _run_sync(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    updates: dict[str, Any] = {}
    if args.s3_bucket:
        updates["s3_bucket"] = args.s3_bucket
    if args.s3_region:
        updates["s3_region"] = args.s3_region
    if args.s3_prefix:
        updates["s3_prefix"] = args.s3_prefix
    if args.data_dir:
        updates["local_data_dir"] = args.data_dir
    if updates:
        config = config.model_copy(update=updates)

    if args.mode == "snapshot-upload":
        if args.force:
            raise SystemExit("--force is only supported with --mode manifest")
        if any([args.all, args.module, args.version, args.submodule]):
            raise SystemExit(
                "--mode snapshot-upload does not accept --all/--module/--version/--submodule"
            )

        s3_service = S3SyncService(config)
        snapshot_results = await s3_service.upload_db_snapshots()

        summary = {
            "mode": args.mode,
            "snapshots": len(snapshot_results),
            "completed": sum(1 for r in snapshot_results if r.success),
            "failed": sum(1 for r in snapshot_results if not r.success),
            "bytes_uploaded": sum(r.size_bytes for r in snapshot_results if r.success),
            "results": [
                {
                    "snapshot_name": r.snapshot_name,
                    "s3_key": r.s3_key,
                    "size_bytes": r.size_bytes,
                    "content_hash": r.content_hash,
                    "success": r.success,
                    "error": r.error,
                }
                for r in snapshot_results
            ],
            "errors": [f"{r.snapshot_name} -> {r.error}" for r in snapshot_results if not r.success and r.error],
        }

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            status = "completed" if summary["failed"] == 0 else "failed"
            print(
                f"Sync status ({args.mode}): {status} "
                f"({summary['completed']}/{summary['snapshots']} uploaded, "
                f"{summary['bytes_uploaded']} bytes)"
            )
            if summary["errors"]:
                print("Errors:")
                for err in summary["errors"]:
                    print(f"- {err}")
        return 0 if summary["failed"] == 0 else 1

    if args.all and any([args.module, args.version, args.submodule]):
        raise SystemExit("--all cannot be combined with --module/--version/--submodule")
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module")
    if not args.all and (not args.module or not args.version):
        raise SystemExit("sync requires --module and --version, or use --all")

    return await _run_manifest_sync(args, config)


async def _run_status(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    if args.data_dir:
        config = config.model_copy(update={"local_data_dir": args.data_dir})
    if args.all and any([args.module, args.version, args.submodule]):
        raise SystemExit("--all cannot be combined with --module/--version/--submodule")
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module")
    if not args.all and (not args.module or not args.version):
        raise SystemExit("status requires --module and --version, or use --all")

    async with SyncOrchestrator(config) as orchestrator:
        if args.all:
            targets = await orchestrator.list_manifest_targets()
            statuses = []
            for target in targets:
                module = target["module"]
                version = target["version"]
                submodule = target["submodule"]
                status = await orchestrator.get_sync_status(
                    module=module,  # type: ignore[arg-type]
                    version=version,  # type: ignore[arg-type]
                    submodule=submodule,  # type: ignore[arg-type]
                )
                statuses.append(status)
            print(json.dumps(statuses, indent=2))
            return 0

        status = await orchestrator.get_sync_status(
            args.module,
            args.version,
            submodule=args.submodule,
        )

    print(json.dumps(status, indent=2))
    return 0


async def _run_index(args: argparse.Namespace) -> int:
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module for index --reindex")

    config = VectorConfig.from_env()
    updates: dict[str, Any] = {}
    if args.data_dir:
        updates["data_dir"] = args.data_dir
    if args.db_filename:
        updates["db_filename"] = args.db_filename
    if args.docs_subdir:
        updates["docs_subdir"] = args.docs_subdir
    if args.chroma_dir:
        updates["chroma_dir"] = args.chroma_dir
    if args.collection:
        updates["collection_name"] = args.collection
    if args.embed_model:
        updates["embedding_model"] = args.embed_model
    if args.max_tokens is not None:
        updates["max_tokens"] = args.max_tokens
    if args.overlap_tokens is not None:
        updates["overlap_tokens"] = args.overlap_tokens
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if updates:
        config = config.model_copy(update=updates)

    reset_count = 0
    deleted_vectors = 0
    if args.reindex:
        storage = StorageManager(config.db_path)
        await storage.connect()
        try:
            module_filter = args.module
            major_filter = args.version
            submodule_filter = args.submodule if module_filter else None
            reset_count = await storage.documents.reset_indexed(
                module=module_filter,
                major=major_filter,
                submodule=submodule_filter,
            )
        finally:
            await storage.close()

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
            before_delete = chroma.count()
            chroma.reset_collection()
            deleted_vectors = before_delete
        else:
            before_delete = chroma.count()
            chroma.delete(where=where_filter)
            after_delete = chroma.count()
            deleted_vectors = max(0, before_delete - after_delete)

    async with VectorIndexer(config=config) as indexer:
        stats = await indexer.index_unindexed(limit=args.limit)

    payload = {
        "documents_indexed": stats.documents_indexed,
        "chunks_indexed": stats.chunks_indexed,
        "documents_reset": reset_count,
        "vectors_deleted": deleted_vectors,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Indexed {payload['documents_indexed']} documents "
            f"({payload['chunks_indexed']} chunks)"
        )
    return 0


async def _run_serve(args: argparse.Namespace) -> int:
    """Run MCP server."""
    from everspring_mcp.mcp.server import create_server
    
    logger = logging.getLogger(__name__)
    
    server = create_server()
    
    if args.json:
        # Output status and continue
        status_info = {
            "status": "starting",
            "transport": args.transport,
            "name": server.name,
        }
        print(json.dumps(status_info, indent=2), file=sys.stderr)
    else:
        print(f"Starting {server.name} MCP server...", file=sys.stderr)
    
    # Initialize and run
    await server.initialize()
    server.run(transport=args.transport)
    
    return 0


async def _run_client(args: argparse.Namespace) -> int:
    """Run interactive client."""
    from everspring_mcp.mcp.client import MCPClient
    
    show_progress = not getattr(args, 'no_progress', False)
    client = MCPClient(show_progress=show_progress)
    
    print("EverSpring MCP - Spring Documentation Search")
    print("=" * 50)
    print("Commands: 'status', 'modules', 'quit', or enter a search query")
    print("Syntax: query [module=X] [version=N] [submodule=X]")
    print()
    
    await client.initialize()
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "status":
            status = await client.get_status()
            print(client.format_status(status))
            continue
        
        if user_input.lower() == "modules":
            modules = await client.list_modules()
            print("Available modules:")
            for mod in modules:
                print(f"  - {mod}")
            continue
        
        # Parse search query with optional filters
        parts = user_input.split()
        query_parts: list[str] = []
        module: str | None = None
        version: int | None = None
        submodule: str | None = None
        
        for part in parts:
            if part.startswith("module="):
                module = part.split("=", 1)[1]
            elif part.startswith("version="):
                try:
                    version = int(part.split("=", 1)[1])
                except ValueError:
                    print(f"Invalid version: {part}")
                    continue
            elif part.startswith("submodule="):
                submodule = part.split("=", 1)[1]
            else:
                query_parts.append(part)
        
        query = " ".join(query_parts)
        if not query:
            print("Please provide a search query")
            continue
        
        # Run search
        response = await client.search(
            query=query,
            module=module,
            version=version,
            submodule=submodule,
        )
        
        print()
        print(client.format_results(response))
    
    return 0


async def _run_search(args: argparse.Namespace) -> int:
    """Run hybrid search command."""
    config = VectorConfig.from_env()
    retriever = HybridRetriever(config=config)
    
    # Build BM25 index if requested or not exists
    if args.build_index or not retriever.ensure_bm25_index():
        print("Building BM25 index...", file=sys.stderr)
        retriever.build_bm25_index()
    
    # Run search
    results = await retriever.search(
        query=args.query,
        top_k=args.top_k,
        module=args.module,
        version_major=args.version,
        deduplicate_urls=not args.no_dedup,
    )
    
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
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print("No results found.")
            return 0
        
        for i, r in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"[{i}] {r.title}")
            print(f"    URL: {r.url}")
            print(f"    Module: {r.module} v{r.version_major}.{r.version_minor}")
            print(f"    Score: {r.score:.4f} (dense: #{r.dense_rank}, sparse: #{r.sparse_rank})")
            print(f"{'='*60}")
            print(r.content)
    
    return 0


async def _run_model_cache(args: argparse.Namespace) -> int:
    """Pre-download embedding model cache for faster first query."""
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
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Model cache ready for {payload['embedding_model']} "
            f"in {payload['elapsed_seconds']:.2f}s"
        )
    return 0


async def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level, args.log_file)

    if args.command == "scrape":
        return await _run_scrape(args)
    if args.command == "sync":
        return await _run_sync(args)
    if args.command == "status":
        return await _run_status(args)
    if args.command == "index":
        return await _run_index(args)
    if args.command == "search":
        return await _run_search(args)
    if args.command == "model-cache":
        return await _run_model_cache(args)
    if args.command == "serve":
        return await _run_serve(args)
    if args.command == "client":
        return await _run_client(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
