"""EverSpring MCP - FastMCP Server Implementation.

Exposes Spring documentation tools via the MCP protocol.
Users interact with this server through the MCPClient.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from ..utils.logging import get_logger
from ..vector.config import VectorConfig
from .models import (
    ProgressNotification,
    SearchParameters,
    SearchResponse,
    StatusResponse,
)
from .tools import SpringDocsTool

logger = get_logger("mcp.server")


class MCPServer:
    """FastMCP-based server for Spring documentation tools.
    
    Exposes:
    - search_spring_docs: Hybrid search with score thresholds
    - get_status: Server/index health status
    - list_modules: Available modules and versions
    
    Progress notifications are sent via the MCP notification system.
    """
    
    def __init__(
        self,
        name: str = "everspring-mcp",
        config: VectorConfig | None = None,
    ) -> None:
        """Initialize MCP server.
        
        Args:
            name: Server name for MCP registration
            config: Vector configuration
        """
        start = time.perf_counter()
        self.name = name
        self.config = config or VectorConfig.from_env()
        
        # Create FastMCP instance
        self._mcp = FastMCP(name)
        
        # Progress notifications queue for async delivery
        self._progress_queue: asyncio.Queue[ProgressNotification] = asyncio.Queue()
        
        # Create search tool with progress callback
        self._tool = SpringDocsTool(
            config=self.config,
            progress_callback=self._on_progress,
        )
        
        # Register MCP tools
        self._register_tools()
        logger.info(f"MCPServer initialized in {time.perf_counter() - start:.2f}s")
    
    def _on_progress(self, notification: ProgressNotification) -> None:
        """Handle progress notification from tool.
        
        Queues notification for async delivery via MCP.
        """
        try:
            self._progress_queue.put_nowait(notification)
        except asyncio.QueueFull:
            logger.warning("Progress queue full, dropping notification")
    
    def _register_tools(self) -> None:
        """Register MCP tools with FastMCP."""
        
        @self._mcp.tool()
        async def search_spring_docs(
            query: str,
            module: str | None = None,
            version: int | None = None,
            submodule: str | None = None,
            top_k: int = 3,
            score_threshold: float = 0.01,
        ) -> dict[str, Any]:
            """Search Spring documentation using hybrid retrieval.
            
            Uses cosine similarity + BM25 with Reciprocal Rank Fusion to find
            the most relevant documentation chunks.
            
            Args:
                query: Natural language query (e.g., "how to configure DataSource")
                module: Filter by Spring module (e.g., "spring-boot", "spring-framework")
                version: Filter by major version (e.g., 4 for Spring Boot 4.x)
                submodule: Filter by submodule (e.g., "redis" for spring-data-redis)
                top_k: Number of results to return (1-10, default 3)
                score_threshold: Minimum relevance score (0.0-1.0, default 0.01)
                
            Returns:
                Search results with status, scores, and documentation content.
                Results below score_threshold are rejected with explanation.
            """
            params = SearchParameters(
                query=query,
                module=module,
                version=version,
                submodule=submodule,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            
            response = await self._tool.search(params)
            return response.model_dump()
        
        @self._mcp.tool()
        async def get_spring_docs_status() -> dict[str, Any]:
            """Get EverSpring MCP server status.
            
            Returns information about:
            - Server health
            - Index readiness
            - Available modules and versions
            - Document counts
            - BM25 index status
            """
            response = await self._tool.get_status()
            return response.model_dump()
        
        @self._mcp.tool()
        async def list_spring_modules() -> list[str]:
            """List available Spring modules.
            
            Returns a list of module names that can be used as filters
            in search_spring_docs (e.g., "spring-boot", "spring-framework").
            """
            return await self._tool.list_available_modules()
    
    @property
    def mcp(self) -> FastMCP:
        """Get the underlying FastMCP instance."""
        return self._mcp
    
    async def initialize(self) -> bool:
        """Initialize server and tools.
        
        Returns:
            True if initialization successful
        """
        logger.info(f"Initializing {self.name} MCP server...")
        return await self._tool.initialize()
    
    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server.
        
        Args:
            transport: Transport type ("stdio" or future options)
        """
        logger.info(f"Starting {self.name} MCP server with {transport} transport")
        self._mcp.run(transport=transport)
    
    async def handle_request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a tool request directly (for testing/client use).
        
        Args:
            tool_name: Name of tool to invoke
            arguments: Tool arguments
            
        Returns:
            Tool response as dict
        """
        if tool_name == "search_spring_docs":
            params = SearchParameters(**arguments)
            response = await self._tool.search(params)
            return response.model_dump()
        
        elif tool_name == "get_spring_docs_status":
            response = await self._tool.get_status()
            return response.model_dump()
        
        elif tool_name == "list_spring_modules":
            modules = await self._tool.list_available_modules()
            return {"modules": modules}
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def get_progress_notifications(
        self,
        timeout: float = 0.1,
    ) -> list[ProgressNotification]:
        """Get any pending progress notifications.
        
        Args:
            timeout: How long to wait for notifications
            
        Returns:
            List of progress notifications (may be empty)
        """
        notifications: list[ProgressNotification] = []
        try:
            while True:
                notification = await asyncio.wait_for(
                    self._progress_queue.get(),
                    timeout=timeout,
                )
                notifications.append(notification)
        except asyncio.TimeoutError:
            pass
        return notifications


def create_server(
    name: str = "everspring-mcp",
    config: VectorConfig | None = None,
) -> MCPServer:
    """Factory function to create MCP server.
    
    Args:
        name: Server name
        config: Vector configuration
        
    Returns:
        Configured MCPServer instance
    """
    return MCPServer(name=name, config=config)


def run_server() -> None:
    """Entry point for running the MCP server standalone."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    
    server = create_server()
    server.run(transport="stdio")


__all__ = ["MCPServer", "create_server", "run_server"]
