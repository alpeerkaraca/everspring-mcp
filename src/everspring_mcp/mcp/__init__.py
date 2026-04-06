"""EverSpring MCP - MCP Server and Client implementation.

This module provides:
- MCPServer: FastMCP-based server exposing Spring documentation tools
- MCPClient: Client interface for interacting with the server
- Tools: Vector search with score thresholds and progress notifications
"""

from .server import MCPServer
from .client import MCPClient
from .tools import SpringDocsTool

__all__ = [
    "MCPServer",
    "MCPClient",
    "SpringDocsTool",
]
