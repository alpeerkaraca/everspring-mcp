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
import logging
from dotenv import load_dotenv

from everspring_mcp.cli import scrape, sync, ingest, index, mcp, model_cache

load_dotenv(dotenv_path="./.env")
logger = logging.getLogger("everspring_mcp")

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

    scrape.add_subparser(subparsers)
    sync.add_subparser(subparsers)
    ingest.add_subparser(subparsers)
    index.add_subparser(subparsers)
    mcp.add_subparser(subparsers)
    model_cache.add_subparser(subparsers)

    return parser

async def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level, args.log_file)

    if hasattr(args, "func"):
        return await args.func(args)

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
