"""EverSpring MCP - MCP Server and Client implementation.

This module provides:
- MCPServer: MCP SDK server exposing Spring documentation tools
- MCPClient: Client interface for interacting with the RAG directly from the terminal
- Tools: Vector search with score thresholds and progress notifications
"""

from everspring_mcp.mcp.server import MCPServer
from everspring_mcp.mcp.client import MCPClient
from everspring_mcp.mcp.tools import SpringDocsTool

__all__ = [
    "MCPServer",
    "MCPClient",
    "SpringDocsTool",
]
