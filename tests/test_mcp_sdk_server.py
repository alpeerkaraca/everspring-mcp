"""Tests for MCP SDK server behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from everspring_mcp.mcp.server import SearchToolArgs, create_server
from everspring_mcp.models.metadata import SearchResult


def _extract_text(result) -> str:
    return "\n".join(
        block.text
        for block in result.content
        if getattr(block, "type", "") == "text"
    )


@pytest.mark.asyncio
async def test_execute_search_tool_formats_markdown() -> None:
    with patch("everspring_mcp.mcp.server.HybridRetriever") as mock_retriever:
        retriever = mock_retriever.return_value
        retriever._embedder.prefetch_model = AsyncMock(return_value=None)
        retriever.ensure_bm25_index.return_value = True
        retriever.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="doc-1",
                    content="Use multiple @Bean methods with requestMatchers.",
                    title="Spring Security Servlet Authorization",
                    url="https://docs.spring.io/security/reference/servlet/authorization/authorize-http-requests.html",
                    module="spring-security",
                    submodule=None,
                    version_major=7,
                    version_minor=0,
                    score=0.1234,
                    dense_rank=1,
                    sparse_rank=2,
                    section_path="Authorize HttpServletRequests",
                    has_code=True,
                ),
            ]
        )

        server = create_server(name="sdk-test")
        result = await server._execute_search_tool(
            SearchToolArgs(query="multiple securityfilterchain beans"),
        )

        assert result.isError is False
        text = _extract_text(result)
        assert "## Spring Docs Search Results" in text
        assert "Spring Security Servlet Authorization" in text
        assert "**Module:** spring-security" in text
        retriever.search.assert_awaited_once_with(
            query="multiple securityfilterchain beans",
            top_k=3,
            module=None,
            version_major=None,
        )


@pytest.mark.asyncio
async def test_execute_search_tool_returns_graceful_error() -> None:
    with patch("everspring_mcp.mcp.server.HybridRetriever") as mock_retriever:
        retriever = mock_retriever.return_value
        retriever._embedder.prefetch_model = AsyncMock(return_value=None)
        retriever.ensure_bm25_index.return_value = True
        retriever.search = AsyncMock(side_effect=RuntimeError("Chroma index missing"))

        server = create_server(name="sdk-test")
        result = await server._execute_search_tool(
            SearchToolArgs(query="multiple securityfilterchain beans"),
        )

        assert result.isError is True
        text = _extract_text(result)
        assert "Search is currently unavailable" in text
        assert "RuntimeError" in text


@pytest.mark.asyncio
async def test_preheat_calls_embedder_prefetch_and_bm25_load() -> None:
    with patch("everspring_mcp.mcp.server.HybridRetriever") as mock_retriever:
        retriever = mock_retriever.return_value
        retriever._embedder.prefetch_model = AsyncMock(return_value=None)
        retriever.ensure_bm25_index.return_value = True

        server = create_server(name="sdk-test")
        await server._preheat_retriever()

        retriever._embedder.prefetch_model.assert_awaited_once()
        retriever.ensure_bm25_index.assert_called_once_with()
        assert server._runtime.preheat_completed is True

