"""EverSpring MCP CLI entrypoint.

Provides commands for:
- Scraping Spring docs into S3
- Syncing knowledge packs or manifests from S3 into local stores
- Indexing local docs into ChromaDB
- Serving MCP tools over stdio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from chromadb.errors import InvalidArgumentError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from everspring_mcp.models.content import ContentType
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.pipeline import PipelineConfig, ScraperPipeline
from everspring_mcp.scraper.registry import SubmoduleRegistry
from everspring_mcp.storage.repository import StorageManager
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.orchestrator import SyncOrchestrator
from everspring_mcp.sync.s3_sync import S3SyncService
from everspring_mcp.vector.chroma_client import ChromaClient
from everspring_mcp.vector.config import VectorConfig, chunk_defaults_for_tier
from everspring_mcp.vector.embeddings import (
    Embedder,
    default_model_for_tier,
)
from everspring_mcp.vector.indexer import VectorIndexer
from everspring_mcp.vector.retriever import HybridRetriever

logger = logging.getLogger("everspring_mcp")
console = Console(stderr=True)


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _render_kv_panel(
    title: str, values: dict[str, Any], *, border_style: str = "cyan"
) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for key, value in values.items():
        grid.add_row(key.replace("_", " ").title(), _format_cell(value))
    console.print(Panel(grid, title=title, border_style=border_style))


def _render_error_table(errors: list[str]) -> None:
    if not errors:
        return
    table = Table(title="Errors", header_style="bold red")
    table.add_column("#", style="bold")
    table.add_column("Message")
    for idx, error in enumerate(errors, 1):
        table.add_row(str(idx), error)
    console.print(table)


def _render_status_rows(title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print(
            Panel("No status data available.", title=title, border_style="yellow")
        )
        return
    preferred_columns = [
        "module",
        "submodule",
        "version",
        "status",
        "files_added",
        "files_modified",
        "files_removed",
        "bytes_downloaded",
    ]
    all_columns = {key for row in rows for key in row}
    columns = [col for col in preferred_columns if col in all_columns]
    if not columns:
        columns = sorted(all_columns)
    table = Table(title=title, header_style="bold blue")
    for column in columns:
        table.add_column(column.replace("_", " ").title())
    for row in rows:
        table.add_row(*[_format_cell(row.get(column)) for column in columns])
    console.print(table)


def _render_search_results(results: list[Any]) -> None:
    if not results:
        console.print(Panel("No results found.", title="Search", border_style="yellow"))
        return

    summary = Table(title="Spring Docs Search Results", header_style="bold magenta")
    summary.add_column("#", style="bold")
    summary.add_column("Title")
    summary.add_column("Module")
    summary.add_column("Version")
    summary.add_column("Score", justify="right")
    summary.add_column("URL", overflow="fold")
    for idx, result in enumerate(results, 1):
        summary.add_row(
            str(idx),
            result.title,
            result.module,
            f"{result.version_major}.{result.version_minor}",
            f"{result.score:.4f}",
            result.url,
        )
    console.print(summary)

    for idx, result in enumerate(results, 1):
        rank_info = (
            f"Dense rank: #{result.dense_rank if result.dense_rank is not None else '-'} | "
            f"Sparse rank: #{result.sparse_rank if result.sparse_rank is not None else '-'}"
        )
        console.print(
            Panel(
                f"{rank_info}\n\n{result.content}",
                title=f"[{idx}] {result.title}",
                subtitle=result.url,
                border_style="cyan",
            )
        )


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


def _model_slug(model_name: str) -> str:
    model_tail = model_name.strip().split("/")[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", model_tail.lower()).strip("-")
    return slug or "model"


def _tier_chroma_dir(tier: str, model_name: str) -> Path:
    return Path.home() / ".everspring" / f"chroma-{tier}-{_model_slug(model_name)}"


def _snapshot_state_path(data_dir: Path) -> Path:
    return data_dir / ".snapshot_state.json"


def _load_snapshot_state(data_dir: Path) -> dict[str, str]:
    state_path = _snapshot_state_path(data_dir)
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    state: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            state[key] = value
    return state


def _save_snapshot_state(data_dir: Path, state: dict[str, str]) -> None:
    state_path = _snapshot_state_path(data_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _restart_current_process() -> None:
    env = os.environ.copy()
    env["EVERSPRING_AUTO_SNAPSHOT_RESTARTED"] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "everspring_mcp.main", *sys.argv[1:]],
        env,
    )


async def _auto_refresh_runtime_snapshots(
    *,
    model_name: str,
    tier: str,
    data_dir: Path,
) -> None:
    if os.environ.get("EVERSPRING_AUTO_SNAPSHOT_RESTARTED") == "1":
        return

    chroma_override = os.environ.get(VectorConfig.ENV_CHROMA_DIR)
    sync_updates: dict[str, Any] = {
        "local_data_dir": data_dir,
        "model_name": model_name,
        "model_tier": tier,
        "chroma_subdir": chroma_override or str(_tier_chroma_dir(tier, model_name)),
    }
    sync_config = SyncConfig.from_env().model_copy(update=sync_updates)
    sync_config.ensure_directories()
    service = S3SyncService(sync_config)
    try:
        selection = await service.find_latest_snapshot_pair(
            model_name=model_name,
            tier=tier,
        )
    except ValueError:
        return

    namespace = sync_config.get_snapshot_namespace(model_name=model_name, tier=tier)
    selected_state = (
        f"{selection.snapshot_token}|{selection.chroma_key}|{selection.sqlite_key}"
    )
    state = _load_snapshot_state(data_dir)
    if state.get(namespace) == selected_state:
        return

    async with SyncOrchestrator(sync_config, s3_service=service) as orchestrator:
        download_result = await orchestrator.download_latest_snapshots(
            model_name=model_name,
            tier=tier,
        )
    if not download_result.success or not download_result.snapshot_token:
        logger.warning(
            "Auto snapshot refresh failed for %s: %s",
            namespace,
            download_result.error or "unknown error",
        )
        return

    state[namespace] = selected_state
    _save_snapshot_state(data_dir, state)
    logger.info(
        "Applied newer snapshot %s for %s; restarting process",
        download_result.snapshot_token,
        namespace,
    )
    _restart_current_process()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EverSpring docs CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Extra log file path",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser(
        "scrape",
        help="Scrape docs to S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scrape.add_argument("--entry-url", default=None, help="Entry URL for discovery")
    scrape.add_argument(
        "--module", default=None, type=_parse_module, help="Spring module"
    )
    scrape.add_argument("--version", default=None, help="Version string (e.g., 4.0.5)")
    scrape.add_argument(
        "--submodule", default=None, help="Optional submodule key (e.g., redis)"
    )
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
    scrape.add_argument(
        "--concurrency", type=int, default=5, help="Max concurrent scrapes"
    )
    scrape.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    scrape.add_argument("--s3-region", default=None, help="S3 region override")
    scrape.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    scrape.add_argument(
        "--no-hash-check", action="store_true", help="Disable hash checks"
    )
    scrape.add_argument("--json", action="store_true", help="Output JSON summary")

    sync = subparsers.add_parser(
        "sync",
        help="Sync docs and snapshots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sync.add_argument(
        "--module", required=False, help="Spring module (e.g., spring-boot)"
    )
    sync.add_argument("--version", required=False, help="Version string (e.g., 4.0.5)")
    sync.add_argument(
        "--submodule", default=None, help="Optional submodule key (e.g., redis)"
    )
    sync.add_argument(
        "--all",
        action="store_true",
        help=(
            "For manifest modes: sync all discovered module/submodule/version targets from "
            "config\\module_submodule_urls.csv. For snapshot-upload: upload snapshots for all "
            "default embedding tiers."
        ),
    )
    sync.add_argument(
        "--mode",
        choices=[
            "manifest",
            "manifest-prime",
            "snapshot-upload",
            "snapshot-download",
        ],
        default="manifest",
        help=(
            "Sync mode (manifest downloads raw docs; manifest-prime pre-generates/uploads "
            "manifest.json files; snapshot-upload backs up local SQLite+Chroma; "
            "snapshot-download restores latest matching SQLite+Chroma snapshots)"
        ),
    )
    sync.add_argument(
        "--force",
        action="store_true",
        help="Force operation in manifest/manifest-prime modes",
    )
    sync.add_argument(
        "--parallel-jobs",
        type=int,
        default=5,
        help="Parallel workers for manifest download/priming",
    )
    sync.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    sync.add_argument("--s3-region", default=None, help="S3 region override")
    sync.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    sync.add_argument(
        "--snapshot-model",
        default=None,
        help="Snapshot namespace model name override for snapshot-upload/download",
    )
    sync.add_argument(
        "--snapshot-tier",
        choices=["main", "slim", "xslim"],
        default=None,
        help="Snapshot namespace tier override for snapshot-upload/download",
    )
    sync.add_argument("--data-dir", default=None, help="Local data directory override")
    sync.add_argument("--json", action="store_true", help="Output JSON summary")

    status = subparsers.add_parser(
        "status",
        help="Show sync status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    status.add_argument(
        "--module", required=False, help="Spring module (e.g., spring-boot)"
    )
    status.add_argument(
        "--version", required=False, help="Version string (e.g., 4.0.5)"
    )
    status.add_argument(
        "--submodule", default=None, help="Optional submodule key (e.g., redis)"
    )
    status.add_argument(
        "--all",
        action="store_true",
        help="Show status for all module/submodule/version entries in local manifest cache",
    )
    status.add_argument(
        "--data-dir", default=None, help="Local data directory override"
    )
    status.add_argument("--json", action="store_true", help="Output JSON summary")

    index = subparsers.add_parser(
        "index",
        help="Index docs into ChromaDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index.add_argument("--limit", type=int, default=50, help="Max documents to index")
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
        "--module", default=None, help="Module filter for --reindex (e.g., spring-boot)"
    )
    index.add_argument(
        "--version", type=int, default=None, help="Major version filter for --reindex"
    )
    index.add_argument(
        "--submodule", default=None, help="Submodule filter for --reindex"
    )
    index.add_argument("--json", action="store_true", help="Output JSON summary")

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
    search.add_argument("--json", action="store_true", help="Output JSON results")

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

    serve = subparsers.add_parser(
        "serve",
        help="Run MCP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    serve.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http"],
        help="MCP transport type",
    )
    serve.add_argument(
        "--tier",
        choices=["main", "slim", "xslim"],
        default="main",
        help="Embedding tier",
    )
    serve.add_argument(
        "--json", action="store_true", help="Output JSON status on startup"
    )

    # Interactive client command
    client_cmd = subparsers.add_parser(
        "client",
        help="Run interactive client",
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
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
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

    if any([args.entry_url, module, version]) and not all(
        [args.entry_url, module, version]
    ):
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
        logger.info(json.dumps(summary, indent=2))
    else:
        logger.info(
            f"Discovery: {summary['discovery']['links']} links "
            f"(duplicates: {summary['discovery']['duplicates']}, "
            f"filtered: {summary['discovery']['filtered']})"
        )
        logger.info(
            f"Scrape results: {summary['results']['success']} success, "
            f"{summary['results']['skipped']} skipped, {summary['results']['failed']} failed"
        )
    return 0 if failed == 0 else 1


def _load_sync_targets_from_matrix(
    matrix_path: Path,
) -> list[tuple[str, str, str | None]]:
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
            targets = _load_sync_targets_from_matrix(
                Path("config") / "module_submodule_urls.csv"
            )
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
                "errors": [
                    f"{r.module}:{r.version} -> {err}"
                    for r in results
                    for err in r.errors
                ],
            }
            if args.json:
                logger.info(json.dumps(summary, indent=2))
            else:
                _render_kv_panel(
                    "Sync Summary (manifest)",
                    {
                        "targets": summary["targets"],
                        "completed": summary["completed"],
                        "failed": summary["failed"],
                        "files_added": summary["files_added"],
                        "files_modified": summary["files_modified"],
                        "files_removed": summary["files_removed"],
                        "bytes_downloaded": summary["bytes_downloaded"],
                    },
                )
                _render_error_table(summary["errors"])
            return 0 if summary["failed"] == 0 else 1

        result = await orchestrator.sync_module(
            module=args.module,
            version=args.version,
            submodule=args.submodule,
            force=args.force,
        )

    if args.json:
        logger.info(result.model_dump_json(indent=2))
    else:
        _render_kv_panel(
            "Sync Status (manifest)",
            {
                "status": result.status.value,
                "module": result.module,
                "version": result.version,
                "submodule": result.submodule or "",
                "files_added": result.files_added,
                "files_modified": result.files_modified,
                "files_removed": result.files_removed,
                "bytes_downloaded": result.bytes_downloaded,
            },
        )
        _render_error_table(result.errors)
    return 0 if result.status.value == "completed" else 1


async def _run_manifest_prime(args: argparse.Namespace, config: SyncConfig) -> int:
    """Generate/upload remote manifest.json files without downloading docs."""
    s3_service = S3SyncService(config)
    parallel_jobs = config.parallel_jobs

    if args.all:
        targets = _load_sync_targets_from_matrix(
            Path("config") / "module_submodule_urls.csv"
        )
    else:
        targets = [(args.module, args.version, args.submodule)]

    semaphore = asyncio.Semaphore(parallel_jobs)

    async def _prime_target(module: str, version: str, submodule: str | None):
        async with semaphore:
            return await s3_service.ensure_manifest(
                module=module,
                version=version,
                submodule=submodule,
                force=args.force,
            )

    tasks = [
        _prime_target(module, version, submodule)
        for module, version, submodule in targets
    ]
    results = await asyncio.gather(*tasks)

    summary = {
        "mode": "manifest-prime",
        "parallel_jobs": parallel_jobs,
        "targets": len(results),
        "uploaded": sum(1 for r in results if r.status == "uploaded"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "files_covered": sum(r.file_count for r in results if r.status != "failed"),
        "total_size_bytes": sum(
            r.total_size_bytes for r in results if r.status != "failed"
        ),
        "results": [r.model_dump() for r in results],
        "errors": [
            f"{r.module}:{r.version}{'/' + r.submodule if r.submodule else ''} -> {r.error}"
            for r in results
            if r.status == "failed" and r.error
        ],
    }

    if args.json:
        logger.info(json.dumps(summary, indent=2, default=str))
    else:
        _render_kv_panel(
            "Manifest Prime Summary",
            {
                "targets": summary["targets"],
                "uploaded": summary["uploaded"],
                "skipped": summary["skipped"],
                "failed": summary["failed"],
                "files_covered": summary["files_covered"],
                "total_size_bytes": summary["total_size_bytes"],
                "parallel_jobs": summary["parallel_jobs"],
            },
        )
        _render_error_table(summary["errors"])
    return 0 if summary["failed"] == 0 else 1


async def _run_sync(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    if args.parallel_jobs < 1:
        raise SystemExit("--parallel-jobs must be >= 1")
    updates: dict[str, Any] = {}
    if args.s3_bucket:
        updates["s3_bucket"] = args.s3_bucket
    if args.s3_region:
        updates["s3_region"] = args.s3_region
    if args.s3_prefix:
        updates["s3_prefix"] = args.s3_prefix
    if args.data_dir:
        updates["local_data_dir"] = Path(args.data_dir)
    snapshot_tier = getattr(args, "snapshot_tier", None)
    snapshot_model = getattr(args, "snapshot_model", None)
    if snapshot_tier:
        updates["model_tier"] = snapshot_tier
    if snapshot_model:
        updates["model_name"] = snapshot_model
    if args.mode in {"manifest", "manifest-prime"}:
        updates["parallel_jobs"] = args.parallel_jobs
        updates["download_concurrency"] = args.parallel_jobs
    if args.mode in {"snapshot-upload", "snapshot-download"} and not (
        args.mode == "snapshot-upload" and args.all
    ):
        effective_snapshot_tier = updates.get("model_tier", config.model_tier)
        effective_snapshot_model = updates.get("model_name", config.model_name)
        snapshot_chroma_override = os.environ.get(VectorConfig.ENV_CHROMA_DIR)
        updates["chroma_subdir"] = snapshot_chroma_override or str(
            _tier_chroma_dir(effective_snapshot_tier, effective_snapshot_model)
        )
    if updates:
        config = config.model_copy(update=updates)

    if args.mode in {"snapshot-upload", "snapshot-download"}:
        if args.force:
            raise SystemExit(
                "--force is only supported with --mode manifest or --mode manifest-prime"
            )
        if any([args.module, args.version, args.submodule]):
            raise SystemExit(
                f"--mode {args.mode} does not accept --module/--version/--submodule"
            )
        if args.mode == "snapshot-download" and args.all:
            raise SystemExit("--all is only supported with --mode snapshot-upload")
        if (
            args.mode == "snapshot-upload"
            and args.all
            and any([snapshot_tier, snapshot_model])
        ):
            raise SystemExit(
                "--all cannot be combined with --snapshot-model/--snapshot-tier for --mode snapshot-upload"
            )

        if args.mode == "snapshot-upload":
            upload_configs: list[SyncConfig] = []
            if args.all:
                for tier_name in ("main", "slim", "xslim"):
                    tier_model = default_model_for_tier(tier_name)
                    upload_configs.append(
                        config.model_copy(
                            update={
                                "model_tier": tier_name,
                                "model_name": tier_model,
                                "chroma_subdir": str(
                                    _tier_chroma_dir(tier_name, tier_model)
                                ),
                            }
                        )
                    )
            else:
                upload_configs.append(config)

            snapshot_results = []
            for upload_config in upload_configs:
                s3_service = S3SyncService(upload_config)
                snapshot_results.extend(
                    await s3_service.upload_db_snapshots(
                        model_name=upload_config.model_name,
                        tier=upload_config.model_tier,
                    )
                )

            summary = {
                "mode": args.mode,
                "scope": "all-tiers" if args.all else "single-namespace",
                "snapshot_namespaces": sorted(
                    {r.snapshot_namespace for r in snapshot_results}
                ),
                "snapshots": len(snapshot_results),
                "completed": sum(1 for r in snapshot_results if r.success),
                "failed": sum(1 for r in snapshot_results if not r.success),
                "bytes_uploaded": sum(
                    r.size_bytes for r in snapshot_results if r.success
                ),
                "results": [
                    {
                        "snapshot_name": r.snapshot_name,
                        "s3_key": r.s3_key,
                        "size_bytes": r.size_bytes,
                        "content_hash": r.content_hash,
                        "snapshot_namespace": r.snapshot_namespace,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in snapshot_results
                ],
                "errors": [
                    f"{r.snapshot_name} -> {r.error}"
                    for r in snapshot_results
                    if not r.success and r.error
                ],
            }

            if args.json:
                logger.info(json.dumps(summary, indent=2))
            else:
                _render_kv_panel(
                    "Snapshot Upload Summary",
                    {
                        "mode": args.mode,
                        "scope": summary["scope"],
                        "snapshots": summary["snapshots"],
                        "completed": summary["completed"],
                        "failed": summary["failed"],
                        "bytes_uploaded": summary["bytes_uploaded"],
                    },
                )
                _render_error_table(summary["errors"])
            return 0 if summary["failed"] == 0 else 1

        if args.mode == "snapshot-download":
            async with SyncOrchestrator(config) as orchestrator:
                snapshot_result = await orchestrator.download_latest_snapshots(
                    model_name=config.model_name,
                    tier=config.model_tier,
                )

            summary = {
                "mode": args.mode,
                "snapshot_namespace": snapshot_result.snapshot_namespace,
                "snapshot_token": snapshot_result.snapshot_token,
                "chroma_snapshot": snapshot_result.chroma_snapshot,
                "sqlite_snapshot": snapshot_result.sqlite_snapshot,
                "bytes_downloaded": snapshot_result.bytes_downloaded,
                "success": snapshot_result.success,
                "error": snapshot_result.error,
            }

            if args.json:
                logger.info(json.dumps(summary, indent=2))
            else:
                if snapshot_result.success:
                    downloaded_mb = snapshot_result.bytes_downloaded / (1024 * 1024)
                    _render_kv_panel(
                        "Snapshot Download Complete",
                        {
                            "namespace": summary["snapshot_namespace"],
                            "snapshot_token": summary["snapshot_token"],
                            "chroma_snapshot": summary["chroma_snapshot"],
                            "sqlite_snapshot": summary["sqlite_snapshot"],
                            "downloaded_mb": round(downloaded_mb, 1),
                        },
                    )
                else:
                    _render_kv_panel(
                        "Snapshot Download Failed",
                        {
                            "mode": args.mode,
                            "error": snapshot_result.error or "unknown error",
                        },
                        border_style="red",
                    )
            return 0 if snapshot_result.success else 1

    if args.all and any([args.module, args.version, args.submodule]):
        raise SystemExit("--all cannot be combined with --module/--version/--submodule")
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module")
    if not args.all and (not args.module or not args.version):
        raise SystemExit("sync requires --module and --version, or use --all")

    if args.mode == "manifest-prime":
        return await _run_manifest_prime(args, config)

    return await _run_manifest_sync(args, config)


async def _run_status(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    if args.data_dir:
        config = config.model_copy(update={"local_data_dir": Path(args.data_dir)})
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
            if args.json:
                logger.info(json.dumps(statuses, indent=2))
            else:
                _render_status_rows("Sync Status (all targets)", statuses)
            return 0

        status = await orchestrator.get_sync_status(
            args.module,
            args.version,
            submodule=args.submodule,
        )

    if args.json:
        logger.info(json.dumps(status, indent=2))
    else:
        _render_status_rows("Sync Status", [status])
    return 0


async def _run_index(args: argparse.Namespace) -> int:
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module for index --reindex")

    config = VectorConfig.from_env()
    updates: dict[str, Any] = {}
    if args.data_dir:
        updates["data_dir"] = Path(args.data_dir)
    if args.db_filename:
        updates["db_filename"] = args.db_filename
    if args.docs_subdir:
        updates["docs_subdir"] = args.docs_subdir
    if args.chroma_dir:
        updates["chroma_dir"] = Path(args.chroma_dir)
    if args.collection:
        updates["collection_name"] = args.collection
    if args.embed_model:
        updates["embedding_model"] = args.embed_model
    selected_tier = getattr(args, "tier", "main")
    updates["embedding_tier"] = selected_tier
    resolved_model = args.embed_model or default_model_for_tier(selected_tier)
    updates["embedding_model"] = resolved_model
    if not args.chroma_dir and not os.environ.get(VectorConfig.ENV_CHROMA_DIR):
        updates["chroma_dir"] = _tier_chroma_dir(selected_tier, resolved_model)
    default_chunk_size, default_overlap = chunk_defaults_for_tier(selected_tier)
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
    if updates:
        config = config.model_copy(update=updates)
    await _auto_refresh_runtime_snapshots(
        model_name=config.embedding_model,
        tier=config.embedding_tier,
        data_dir=config.data_dir,
    )

    reset_count = 0
    deleted_vectors = 0
    logger.info(f"Starting indexing with config: {config.model_dump_json(indent=2)}")
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

    bm25_index_built = False
    if args.build_bm25 and config.embedding_tier != "main":
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
        "bm25_index_path": str(config.data_dir / "bm25_index.pkl")
        if bm25_index_built
        else None,
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


async def _run_serve(args: argparse.Namespace) -> int:
    """Run MCP server."""
    from everspring_mcp.mcp.server import create_server

    config = VectorConfig.from_env()
    selected_tier = getattr(args, "tier", "main")
    resolved_model = default_model_for_tier(selected_tier)
    updates: dict[str, Any] = {
        "embedding_tier": selected_tier,
        "embedding_model": resolved_model,
    }
    if not os.environ.get(VectorConfig.ENV_CHROMA_DIR):
        updates["chroma_dir"] = _tier_chroma_dir(selected_tier, resolved_model)
    config = config.model_copy(update=updates)
    await _auto_refresh_runtime_snapshots(
        model_name=config.embedding_model,
        tier=config.embedding_tier,
        data_dir=config.data_dir,
    )
    server = create_server(config=config)

    if args.json:
        # Output status and continue
        status_info = {
            "status": "starting",
            "transport": args.transport,
            "name": server.name,
        }
        logger.info(json.dumps(status_info, indent=2))
    else:
        logger.info(f"Starting {server.name} MCP server...")

    if args.transport == "stdio":
        logger.info("Using stdio transport")
        await server.serve_stdio()
    elif args.transport == "http":
        logger.info("Using HTTP transport")
        await server.serve_http()
    else:
        raise SystemExit(f"Unsupported transport: {args.transport}")

    return 0


async def _run_client(args: argparse.Namespace) -> int:
    """Run interactive client."""
    from everspring_mcp.mcp.terminal_search import LocalSearchCLI

    show_progress = not getattr(args, "no_progress", False)
    client = LocalSearchCLI(show_progress=show_progress)

    logger.info("EverSpring MCP - Spring Documentation Search")
    logger.info("=" * 50)
    logger.info("Commands: 'status', 'modules', 'quit', or enter a search query")
    logger.info("Syntax: query [module=X] [version=N] [submodule=X]")
    logger.info("")

    await client.initialize()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            logger.info("Goodbye!")
            break

        if user_input.lower() == "status":
            status = await client.get_status()
            logger.info(client.format_status(status))
            continue

        if user_input.lower() == "modules":
            modules = await client.list_modules()
            logger.info("Available modules:")
            for mod in modules:
                logger.info(f"  - {mod}")
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
                    logger.info(f"Invalid version: {part}")
                    continue
            elif part.startswith("submodule="):
                submodule = part.split("=", 1)[1]
            else:
                query_parts.append(part)

        query = " ".join(query_parts)
        if not query:
            logger.info("Please provide a search query")
            continue

        # Run search
        response = await client.search(
            query=query,
            module=module,
            version=version,
            submodule=submodule,
        )

        logger.info("")
        logger.info(client.format_results(response))

    return 0


async def _run_search(args: argparse.Namespace) -> int:
    """Run hybrid search command."""
    config = VectorConfig.from_env()
    selected_tier = getattr(args, "tier", "main")
    resolved_model = default_model_for_tier(selected_tier)
    updates: dict[str, Any] = {
        "embedding_tier": selected_tier,
        "embedding_model": resolved_model,
    }
    if not os.environ.get(VectorConfig.ENV_CHROMA_DIR):
        updates["chroma_dir"] = _tier_chroma_dir(selected_tier, resolved_model)
    config = config.model_copy(update=updates)
    await _auto_refresh_runtime_snapshots(
        model_name=config.embedding_model,
        tier=config.embedding_tier,
        data_dir=config.data_dir,
    )
    retriever = HybridRetriever(config=config)

    # Build BM25 index for non-main tiers if requested or not exists
    if config.embedding_tier != "main" and (
        args.build_index or not retriever.ensure_bm25_index()
    ):
        logger.info("Building BM25 index...")
        retriever.build_bm25_index()

    # Run search
    try:
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
        _render_search_results(results)

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
