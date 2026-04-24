from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from everspring_mcp.models.content import ContentType
from everspring_mcp.scraper.pipeline import PipelineConfig, ScraperPipeline
from everspring_mcp.scraper.registry import SubmoduleRegistry
from everspring_mcp.cli.utils import _parse_module, _parse_content_type, _parse_version

logger = logging.getLogger("everspring_mcp")

async def _run_scrape(args: argparse.Namespace) -> int:
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


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
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
    scrape.set_defaults(func=_run_scrape)
