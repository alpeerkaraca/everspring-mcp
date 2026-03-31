"""EverSpring MCP CLI entrypoint.

Provides commands for:
- Scraping Spring docs into S3
- Syncing S3 docs into local SQLite cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.pipeline import PipelineConfig, ScrapeTarget, ScraperPipeline
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.orchestrator import SyncOrchestrator


def _parse_module(value: str) -> SpringModule:
    """Parse module string to SpringModule enum."""
    normalized = value.strip().lower()
    for module in SpringModule:
        if module.value == normalized:
            return module
    raise argparse.ArgumentTypeError(f"Invalid module: {value}")


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
    scrape.add_argument("--entry-url", required=True, help="Entry URL for discovery")
    scrape.add_argument("--module", required=True, type=_parse_module, help="Spring module")
    scrape.add_argument("--version", required=True, help="Version string (e.g., 4.0.5)")
    scrape.add_argument("--concurrency", type=int, default=5, help="Max concurrent scrapes")
    scrape.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    scrape.add_argument("--s3-region", default=None, help="S3 region override")
    scrape.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    scrape.add_argument("--no-hash-check", action="store_true", help="Disable hash checks")

    sync = subparsers.add_parser(
        "sync",
        help="Sync docs from S3 into local SQLite cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sync.add_argument("--module", required=True, help="Spring module (e.g., spring-boot)")
    sync.add_argument("--version", required=True, help="Version string (e.g., 4.0.5)")
    sync.add_argument("--force", action="store_true", help="Force sync even if manifest unchanged")
    sync.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    sync.add_argument("--s3-region", default=None, help="S3 region override")
    sync.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    sync.add_argument("--data-dir", default=None, help="Local data directory override")

    status = subparsers.add_parser(
        "status",
        help="Show local sync status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    status.add_argument("--module", required=True, help="Spring module (e.g., spring-boot)")
    status.add_argument("--version", required=True, help="Version string (e.g., 4.0.5)")
    status.add_argument("--data-dir", default=None, help="Local data directory override")

    return parser


def _configure_logging(level: str, log_file: str | None) -> None:
    """Configure logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )


async def _run_scrape(args: argparse.Namespace) -> int:
    # Avoid requiring env vars when overrides are provided
    if args.s3_bucket or args.s3_region or args.s3_prefix:
        config = PipelineConfig(
            s3_bucket=args.s3_bucket or "everspring-mcp-kb",
            aws_region=args.s3_region or "eu-central-1",
            s3_prefix=args.s3_prefix or "docs",
            enable_hash_check=not args.no_hash_check,
        )
    else:
        config = PipelineConfig.from_env()
    if args.no_hash_check:
        config = config.model_copy(update={"enable_hash_check": False})

    module = args.module
    version = _parse_version(args.version, module)

    pipeline = ScraperPipeline(config)
    discovery_result, scrape_results = await pipeline.discover_and_scrape(
        entry_url=args.entry_url,
        module=module,
        version=version,
        concurrency=args.concurrency,
    )

    success = sum(1 for r in scrape_results if r.status.value == "success")
    skipped = sum(1 for r in scrape_results if r.status.value == "skipped")
    failed = sum(1 for r in scrape_results if r.status.value == "failed")

    print(
        f"Discovery: {discovery_result.link_count} links "
        f"(duplicates: {discovery_result.duplicates_removed}, filtered: {discovery_result.filtered_out})"
    )
    print(f"Scrape results: {success} success, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


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

    async with SyncOrchestrator(config) as orchestrator:
        result = await orchestrator.sync_module(
            module=args.module,
            version=args.version,
            force=args.force,
        )

    print(
        f"Sync status: {result.status.value} "
        f"(+{result.files_added} ~{result.files_modified} -{result.files_removed}, "
        f"{result.bytes_downloaded} bytes)"
    )
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"- {err}")
    return 0 if result.status.value == "completed" else 1


async def _run_status(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    if args.data_dir:
        config = config.model_copy(update={"local_data_dir": args.data_dir})

    async with SyncOrchestrator(config) as orchestrator:
        status = await orchestrator.get_sync_status(args.module, args.version)

    print(json.dumps(status, indent=2))
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
