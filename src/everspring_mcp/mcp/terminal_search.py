"""EverSpring MCP - Client for interacting with the RAG server.

Provides a high-level interface for searching Spring documentation
with progress tracking and result formatting.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import TextIO

from everspring_mcp.mcp.models import (
    ProgressNotification,
    SearchParameters,
    SearchResponse,
    SearchStatus,
    StatusResponse,
)
from everspring_mcp.mcp.server import MCPServer
from everspring_mcp.utils.logging import get_logger
from everspring_mcp.vector.config import VectorConfig

logger = get_logger("mcp.terminal_search")


class LocalSearchCLI:
    """Client for interacting with EverSpring RAG server.

    Provides:
    - High-level search interface with progress tracking
    - Result formatting for display
    - Module/version discovery

    Users should interact via this client rather than directly with server.
    """

    def __init__(
        self,
        config: VectorConfig | None = None,
        progress_output: TextIO | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize MCP client.

        Args:
            config: Vector configuration
            progress_output: Stream for progress messages (default: stderr)
            show_progress: Whether to display progress notifications
        """
        start = time.perf_counter()
        self.config = config or VectorConfig.from_env()
        self._progress_output = progress_output or sys.stderr
        self._show_progress = show_progress

        # Create server instance (runs in-process)
        self._server = MCPServer(config=self.config)
        self._initialized = False
        logger.info(f"MCPClient initialized in {time.perf_counter() - start:.2f}s")

    def _print_progress(self, notification: ProgressNotification) -> None:
        """Print progress notification to output stream."""
        if not self._show_progress:
            return

        bar_width = 30
        filled = int(notification.percentage / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        msg = f"\r[{bar}] {notification.percentage:5.1f}% | {notification.stage}: {notification.message}"
        self._progress_output.write(msg)
        self._progress_output.flush()

        if notification.percentage >= 100:
            self._progress_output.write("\n")
            self._progress_output.flush()

    async def initialize(self) -> bool:
        """Initialize client and server.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        # Set progress callback
        self._server._tool._progress_callback = self._print_progress

        success = await self._server.initialize()
        self._initialized = success
        return success

    async def search(
        self,
        query: str,
        module: str | None = None,
        version: int | None = None,
        submodule: str | None = None,
        top_k: int = 3,
        score_threshold: float = 0.01,
    ) -> SearchResponse:
        """Search Spring documentation.

        Args:
            query: Natural language search query
            module: Filter by module (e.g., "spring-boot")
            version: Filter by major version (e.g., 4)
            submodule: Filter by submodule (e.g., "redis")
            top_k: Number of results (1-10)
            score_threshold: Minimum relevance score

        Returns:
            SearchResponse with results and status
        """
        if not self._initialized:
            await self.initialize()

        params = SearchParameters(
            query=query,
            module=module,
            version=version,
            submodule=submodule,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        return await self._server._tool.search(params)

    async def get_status(self) -> StatusResponse:
        """Get server and index status.

        Returns:
            StatusResponse with health information
        """
        if not self._initialized:
            await self.initialize()

        return await self._server._tool.get_status()

    async def list_modules(self) -> list[str]:
        """Get list of available modules.

        Returns:
            List of module names
        """
        if not self._initialized:
            await self.initialize()

        return await self._server._tool.list_available_modules()

    def format_results(
        self,
        response: SearchResponse,
        max_content_length: int = 500,
    ) -> str:
        """Format search response for display.

        Args:
            response: Search response to format
            max_content_length: Maximum content preview length

        Returns:
            Formatted string for display
        """
        lines: list[str] = []

        # Status line
        status_icons = {
            SearchStatus.SUCCESS: "✓",
            SearchStatus.NO_RESULTS: "○",
            SearchStatus.BELOW_THRESHOLD: "⚠",
            SearchStatus.ERROR: "✗",
        }
        icon = status_icons.get(response.status, "?")
        lines.append(f"{icon} {response.message}")
        lines.append("")

        # Filters applied
        if response.filters_applied:
            filter_strs = [f"{k}={v}" for k, v in response.filters_applied.items()]
            lines.append(f"Filters: {', '.join(filter_strs)}")

        # Result stats
        lines.append(
            f"Found: {response.results_found} | "
            f"Returned: {response.results_returned} | "
            f"Threshold: {response.score_threshold}"
        )
        lines.append("")

        # Results
        for result in response.results:
            lines.append("=" * 60)
            lines.append(f"[{result.rank}] {result.title}")
            lines.append(f"    URL: {result.url}")

            mod_version = f"{result.module} {result.version}"
            if result.submodule:
                mod_version = f"{result.module}/{result.submodule} {result.version}"
            lines.append(f"    Module: {mod_version}")

            lines.append(f"    Score: {result.score:.4f} ({result.score_breakdown})")

            if result.has_code:
                lines.append("    Contains: Code examples")

            lines.append("=" * 60)

            # Content preview
            content = result.content
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            lines.append(content)
            lines.append("")

        # Warnings
        if response.warnings:
            lines.append("Warnings:")
            for warning in response.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)

    def format_status(self, status: StatusResponse) -> str:
        """Format status response for display.

        Args:
            status: Status response to format

        Returns:
            Formatted string
        """
        lines: list[str] = []

        health = "✓ Healthy" if status.healthy else "✗ Unhealthy"
        lines.append(f"Server Status: {health}")

        index = "Ready" if status.index_ready else "Not Ready"
        lines.append(f"Index Status: {index}")

        lines.append(f"Total Documents: {status.total_documents}")

        bm25 = "Loaded" if status.bm25_index_loaded else "Not Loaded"
        lines.append(f"BM25 Index: {bm25}")

        if status.modules:
            lines.append("")
            lines.append("Available Modules:")
            for mod in status.modules:
                versions = ", ".join(str(v) for v in mod.versions)
                submodules = ""
                if mod.submodules:
                    submodules = f" (submodules: {', '.join(mod.submodules)})"
                lines.append(
                    f"  - {mod.name}: versions [{versions}], "
                    f"{mod.doc_count} docs{submodules}"
                )

        return "\n".join(lines)


async def interactive_search() -> None:
    """Run interactive search session."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    client = MCPClient(show_progress=True)

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
            logger.info("Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            logger.info("Goodbye!")
            break

        if user_input.lower() == "status":
            status = await client.get_status()
            logger.info("%s", client.format_status(status))
            continue

        if user_input.lower() == "modules":
            modules = await client.list_modules()
            logger.info("Available modules:")
            for mod in modules:
                logger.info("  - %s", mod)
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
                    logger.warning("Invalid version: %s", part)
                    continue
            elif part.startswith("submodule="):
                submodule = part.split("=", 1)[1]
            else:
                query_parts.append(part)

        query = " ".join(query_parts)
        if not query:
            logger.warning("Please provide a search query")
            continue

        # Run search
        response = await client.search(
            query=query,
            module=module,
            version=version,
            submodule=submodule,
        )

        logger.info("")
        logger.info("%s", client.format_results(response))


def run_interactive() -> None:
    """Entry point for interactive client."""
    asyncio.run(interactive_search())


__all__ = ["MCPClient", "interactive_search", "run_interactive"]
