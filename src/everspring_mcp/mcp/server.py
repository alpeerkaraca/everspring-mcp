"""EverSpring MCP - MCP SDK server implementation."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from everspring_mcp.mcp.models import (
    ProgressNotification,
    SearchParameters,
    SearchStatus,
    StructuredErrorResponse,
)
from everspring_mcp.mcp.prompt import PromptBuilder
from everspring_mcp.mcp.tools import SpringDocsTool
from everspring_mcp.utils.logging import get_logger
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.retriever import HybridRetriever

logger = get_logger("mcp.server")

TOOL_NAME = "search_spring_docs"
TOOL_DESCRIPTION = (
    "Searches the Spring ecosystem documentation using a Hybrid RAG (Vector + "
    "Keyword) approach. Returns exact code snippets and architectural explanations."
)
SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural language search query.",
            "minLength": 1,
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return.",
            "default": 3,
            "minimum": 1,
            "maximum": 10,
        },
        "module": {
            "type": ["string", "null"],
            "description": "Optional module filter (e.g., spring-boot).",
        },
        "version_major": {
            "type": ["integer", "null"],
            "description": "Optional major version filter (e.g., 4).",
            "minimum": 1,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


class SearchToolArgs(BaseModel):
    """Input model for search_spring_docs tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    module: str | None = None
    version_major: int | None = Field(default=None, ge=1)


@dataclass
class _RuntimeState:
    """Runtime state for MCP SDK handlers."""

    retriever: HybridRetriever | None = None
    preheat_task: asyncio.Task[None] | None = None
    preheat_error: Exception | None = None
    preheat_completed: bool = False


class MCPServer:
    """MCP SDK server exposing Spring documentation search."""

    def __init__(
        self,
        name: str = "everspring-mcp",
        config: VectorConfig | None = None,
    ) -> None:
        start = time.perf_counter()
        self.name = name
        self.config = config or VectorConfig.from_env()

        self._runtime = _RuntimeState()
        self._progress_queue: asyncio.Queue[ProgressNotification] = asyncio.Queue()

        # Compatibility surface for existing in-process client and tests.
        self._tool = SpringDocsTool(
            config=self.config,
            progress_callback=self._on_progress,
        )

        # MCP SDK server with explicit lifecycle hook.
        self._server: Server[Any, Any] = Server(
            name,
            version="0.1.0",
            lifespan=self._lifespan,
        )
        self._register_tools()
        logger.info(f"MCPServer initialized in {time.perf_counter() - start:.2f}s")

    def _on_progress(self, notification: ProgressNotification) -> None:
        """Queue compatibility progress notifications."""
        try:
            self._progress_queue.put_nowait(notification)
        except asyncio.QueueFull:
            logger.warning("Progress queue full, dropping notification")

    def _ensure_retriever(self) -> HybridRetriever:
        """Get or create the runtime retriever without forcing model load."""
        if self._runtime.retriever is None:
            self._runtime.retriever = HybridRetriever(self.config)
        return self._runtime.retriever

    def _start_preheat(self) -> None:
        """Start asynchronous preheating exactly once per server process."""
        if self._runtime.preheat_completed:
            return
        if self._runtime.preheat_error is not None:
            return
        task = self._runtime.preheat_task
        if task is not None and not task.done():
            return

        loop = asyncio.get_running_loop()
        self._runtime.preheat_task = loop.create_task(
            self._preheat_retriever(),
            name="everspring-mcp-preheat",
        )

    async def _await_preheat_completion(self) -> None:
        """Wait for preheat if it is in progress."""
        task = self._runtime.preheat_task
        if task is None:
            return
        if task.done():
            return
        logger.info("Waiting for startup preheating to complete before serving search")
        await asyncio.shield(task)

    async def _preheat_retriever(self) -> None:
        """Preheat model + BM25 right after server initialization."""
        retriever = self._ensure_retriever()
        started_at = datetime.now(UTC).isoformat()
        logger.info("Server preheating started at %s", started_at)
        try:
            await retriever._embedder.prefetch_model()
            bm25_loaded = retriever.ensure_bm25_index()
            self._runtime.preheat_completed = True
            completed_at = datetime.now(UTC).isoformat()
            logger.info(
                "Server preheating completed at %s (bm25_loaded=%s)",
                completed_at,
                bm25_loaded,
            )
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            self._runtime.preheat_error = exc
            logger.error("Server preheating failed: %s", exc)

    @asynccontextmanager
    async def _lifespan(self, _: Server[Any, Any]):
        """Server lifecycle: start quickly, then warm heavy components."""
        logger.info("MCP server initialized at %s", datetime.now(UTC).isoformat())
        self._ensure_retriever()
        self._start_preheat()
        try:
            yield {"started_at": datetime.now(UTC).isoformat()}
        finally:
            task = self._runtime.preheat_task
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    @staticmethod
    def _tool_definition() -> types.Tool:
        return types.Tool(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            inputSchema=SEARCH_TOOL_SCHEMA,
        )

    def _register_tools(self) -> None:
        """Register MCP SDK list_tools and call_tool handlers."""

        @self._server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            return [self._tool_definition()]

        @self._server.call_tool(validate_input=True)
        async def _call_tool(
            tool_name: str,
            arguments: dict[str, Any],
        ) -> types.CallToolResult:
            if tool_name != TOOL_NAME:
                return self._error_result(
                    f"Unknown tool '{tool_name}'. Available tool: {TOOL_NAME}.",
                )

            try:
                params = SearchToolArgs.model_validate(arguments)
            except ValidationError as exc:
                return self._error_result(f"Invalid arguments: {exc}")

            return await self._execute_search_tool(params)

    @staticmethod
    def _format_runtime_error(error: Exception) -> StructuredErrorResponse:
        return StructuredErrorResponse(
            error_type="runtime_unavailable",
            message=(
                "Search is currently unavailable because local retrieval data is "
                "missing or not readable."
            ),
            resolution_hints=[
                "Run local sync/index flows (including BM25 build).",
                "Verify Chroma and SQLite snapshot files exist and are readable.",
                "Retry the same query after local data is restored.",
            ],
            context={
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            },
        )

    @staticmethod
    def _serialize_structured_error(error: StructuredErrorResponse) -> str:
        """Serialize structured error payload for MCP text transport."""
        return error.model_dump_json(indent=2)

    @classmethod
    def _error_result(
        cls, message: str | StructuredErrorResponse
    ) -> types.CallToolResult:
        payload = (
            cls._serialize_structured_error(message)
            if isinstance(message, StructuredErrorResponse)
            else message
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=payload)],
            isError=True,
        )

    async def _execute_search_tool(
        self, params: SearchToolArgs
    ) -> types.CallToolResult:
        """Execute search_spring_docs tool call."""
        self._ensure_retriever()
        self._start_preheat()
        await self._await_preheat_completion()

        if self._runtime.preheat_error is not None:
            return self._error_result(
                self._format_runtime_error(self._runtime.preheat_error),
            )

        try:
            # Validate module if provided
            status = await self._tool.get_status()
            available_modules = [module.name for module in status.modules]
            if (
                params.module
                and available_modules
                and params.module not in available_modules
            ):
                return self._error_result(
                    StructuredErrorResponse(
                        error_type="invalid_module",
                        message=(
                            f"Module '{params.module}' is not available in the current index."
                        ),
                        resolution_hints=[
                            "Use one of the available modules from context.available_modules.",
                            "Run sync/index to refresh local data if the module should exist.",
                            "Retry without module filter to inspect broader results.",
                        ],
                        context={
                            "requested_module": params.module,
                            "available_modules": available_modules,
                            "available_versions_by_module": {
                                m.name: m.versions for m in status.modules
                            },
                        },
                    )
                )

            search_params = SearchParameters(
                query=params.query,
                top_k=params.top_k,
                module=params.module,
                version=params.version_major,
            )
            response = await self._tool.search(search_params)

            if response.status == SearchStatus.ERROR:
                return self._error_result(
                    StructuredErrorResponse(
                        error_type="search_error",
                        message=response.message,
                        resolution_hints=[
                            "Check module/version filters for typos or unsupported values.",
                            "Run local sync/index flows to refresh retrieval data.",
                            "Retry with a broader query and fewer filters.",
                        ],
                        context={
                            "query": params.query,
                            "module": params.module,
                            "version_major": params.version_major,
                        },
                    )
                )

            builder = (
                PromptBuilder()
                .add_system_prompt(f"Status: {response.message}")
                .add_user_query(params.query)
                .add_filters(module=params.module, version=params.version_major)
                .add_retrieved_context(response.results)
            )

            return types.CallToolResult(
                content=[types.TextContent(type="text", text=builder.build())],
                isError=False,
            )
        except Exception as exc:
            logger.error("search_spring_docs failed: %s", exc)
            return self._error_result(self._format_runtime_error(exc))

    @property
    def mcp(self) -> Server[Any, Any]:
        """Get the underlying MCP SDK Server instance."""
        return self._server

    async def initialize(self) -> bool:
        """Compatibility init for in-process MCPClient usage."""
        logger.info("Initializing in-process MCP server compatibility layer")
        self._start_preheat()
        return await self._tool.initialize()

    async def serve_stdio(self) -> None:
        """Run MCP SDK server over stdio transport."""
        logger.info("Starting %s MCP SDK server over stdio", self.name)
        self._ensure_retriever()
        self._start_preheat()
        await self.initialize()

        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(
                    notification_options=NotificationOptions(),
                ),
            )

    async def serve_http(self) -> None:
        """Run MCP SDK server over HTTP transport."""
        logger.info("Starting %s MCP SDK server over HTTP", self.name)
        self._ensure_retriever()
        self._start_preheat()
        await self.initialize()
        from everspring_mcp.http.serve_http import serve_http_via_granian

        os.environ[VectorConfig.ENV_EMBED_TIER] = self.config.embedding_tier
        os.environ[VectorConfig.ENV_EMBED_MODEL] = self.config.embedding_model
        os.environ[VectorConfig.ENV_CHROMA_DIR] = str(self.config.chroma_dir)
        os.environ[VectorConfig.ENV_DATA_DIR] = str(self.config.data_dir)

        exit_code = await serve_http_via_granian()
        if exit_code != 0:
            raise RuntimeError(f"Granian exited with status code {exit_code}")

    def run(self, transport: str = "stdio") -> None:
        """Synchronous wrapper for stdio/http MCP serving."""
        if transport == "stdio":
            asyncio.run(self.serve_stdio())
            return
        if transport == "http":
            asyncio.run(self.serve_http())
            return
        raise ValueError(f"Unsupported transport: {transport}")

    async def handle_request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Compatibility request handler for existing tests and MCPClient."""
        if tool_name == TOOL_NAME:
            params = SearchParameters(**arguments)
            response = await self._tool.search(params)
            return response.model_dump()
        if tool_name == "get_spring_docs_status":
            response = await self._tool.get_status()
            return response.model_dump()
        if tool_name == "list_spring_modules":
            modules = await self._tool.list_available_modules()
            return {"modules": modules}
        raise ValueError(f"Unknown tool: {tool_name}")

    async def get_progress_notifications(
        self,
        timeout: float = 0.1,
    ) -> list[ProgressNotification]:
        """Get any pending progress notifications from compatibility tool path."""
        notifications: list[ProgressNotification] = []
        try:
            while True:
                notification = await asyncio.wait_for(
                    self._progress_queue.get(),
                    timeout=timeout,
                )
                notifications.append(notification)
        except TimeoutError:
            pass
        return notifications


def create_server(
    name: str = "everspring-mcp",
    config: VectorConfig | None = None,
) -> MCPServer:
    """Create configured MCP server instance."""
    return MCPServer(name=name, config=config)


def run_server() -> None:
    """Entrypoint for running the MCP server standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    server = create_server()
    server.run(transport="stdio")


__all__ = ["MCPServer", "create_server", "run_server"]
