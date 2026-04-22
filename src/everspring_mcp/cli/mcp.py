from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import default_model_for_tier
from everspring_mcp.cli.utils import (
    _tier_chroma_dir,
    _auto_refresh_runtime_snapshots,
    _parse_positive_int,
)

logger = logging.getLogger("everspring_mcp")

async def _run_serve(args: argparse.Namespace) -> int:
    from everspring_mcp.http.serve_http import serve_http_via_granian
    from everspring_mcp.mcp.server import create_server

    for option_name in ("workers", "backlog", "threads"):
        option_value = getattr(args, option_name, None)
        if option_value is not None and option_value <= 0:
            raise SystemExit(f"--{option_name} must be greater than 0")

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
    if args.json:
        status_info = {
            "status": "starting",
            "transport": args.transport,
            "name": "everspring-mcp",
            "workers": args.workers,
            "backlog": args.backlog,
            "threads": args.threads,
        }
        logger.info(json.dumps(status_info, indent=2))
    else:
        logger.info("Starting everspring-mcp MCP server...")

    if args.transport == "stdio":
        if (
            args.workers is not None
            or args.backlog is not None
            or args.threads is not None
        ):
            raise SystemExit(
                "--workers/--backlog/--threads can only be used with --transport http"
            )
        logger.info("Using stdio transport")
        server = create_server(config=config)
        await server.serve_stdio()
    elif args.transport == "http":
        logger.info("Using HTTP transport")
        env = os.environ.copy()
        env[VectorConfig.ENV_EMBED_TIER] = config.embedding_tier
        env[VectorConfig.ENV_EMBED_MODEL] = config.embedding_model
        env[VectorConfig.ENV_CHROMA_DIR] = str(config.chroma_dir)
        env[VectorConfig.ENV_DATA_DIR] = str(config.data_dir)
        return await serve_http_via_granian(
            workers=args.workers,
            backlog=args.backlog,
            threads=args.threads,
            log_level=str(args.log_level),
            env=env,
        )
    else:
        raise SystemExit(f"Unsupported transport: {args.transport}")

    return 0

async def _run_client(args: argparse.Namespace) -> int:
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

        response = await client.search(
            query=query,
            module=module,
            version=version,
            submodule=submodule,
        )

        logger.info("")
        logger.info(client.format_results(response))

    return 0


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
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
    serve.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        help="Granian worker processes (HTTP transport only)",
    )
    serve.add_argument(
        "--backlog",
        type=_parse_positive_int,
        default=None,
        help="Granian socket backlog (HTTP transport only)",
    )
    serve.add_argument(
        "--threads",
        type=_parse_positive_int,
        default=None,
        help="Granian runtime threads per worker (HTTP transport only)",
    )
    serve.set_defaults(func=_run_serve)

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
    client_cmd.set_defaults(func=_run_client)
