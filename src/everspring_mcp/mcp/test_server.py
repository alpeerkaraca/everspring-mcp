"""Lightweight MCP test server - no heavy dependencies.

Run with: uv run mcp dev src/everspring_mcp/mcp/test_server.py:mcp
"""

from mcp.server.fastmcp import FastMCP

# Create simple FastMCP instance
mcp = FastMCP("everspring-test")


@mcp.tool()
def ping() -> str:
    """Simple ping test."""
    return "pong"


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return f"Echo: {message}"


@mcp.tool()
def search_docs(query: str, module: str | None = None) -> dict:
    """Mock search - returns dummy data."""
    return {
        "status": "success",
        "query": query,
        "module": module,
        "results": [
            {
                "title": "Test Result",
                "url": "https://docs.spring.io/test",
                "score": 0.95,
                "content": f"Mock result for query: {query}",
            }
        ],
    }
