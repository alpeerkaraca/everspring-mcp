"""Tests for MCP SDK server behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from everspring_mcp.mcp.models import SearchStatus
from everspring_mcp.mcp.server import SearchToolArgs, create_server


def _extract_text(result) -> str:
    return "\n".join(
        block.text for block in result.content if getattr(block, "type", "") == "text"
    )


@pytest.mark.asyncio
async def test_execute_search_tool_formats_markdown() -> None:
    with patch("everspring_mcp.mcp.server.HybridRetriever") as mock_retriever:
        retriever = mock_retriever.return_value
        retriever._embedder.prefetch_model = AsyncMock(return_value=None)
        retriever.ensure_bm25_index.return_value = True

        server = create_server(name="sdk-test")
        server._tool.get_status = AsyncMock(  # type: ignore[method-assign]
            return_value=type("_FakeStatus", (), {"modules": []})()
        )
        server._tool.search = AsyncMock(  # type: ignore[method-assign]
            return_value=type(
                "_SearchResponse",
                (),
                {
                    "status": SearchStatus.SUCCESS,
                    "message": "Found 1 relevant results",
                    "results": [
                        type(
                            "_ResultItem",
                            (),
                            {
                                "content": "\n".join(
                                    [
                                        "### 1. Spring Security Servlet Authorization",
                                        "- **URL:** https://docs.spring.io/security/reference/servlet/authorization/authorize-http-requests.html",
                                        "- **Module:** spring-security",
                                        "- **Version:** v7.0",
                                        "- **Score:** 0.1234",
                                        "",
                                        "Use multiple @Bean methods with requestMatchers.",
                                    ]
                                )
                            },
                        )()
                    ],
                },
            )()
        )
        result = await server._execute_search_tool(
            SearchToolArgs(query="multiple securityfilterchain beans"),
        )

        assert result.isError is False
        text = _extract_text(result)
        assert "## Spring Docs Search Results" in text
        assert "Spring Security Servlet Authorization" in text
        assert "**Module:** spring-security" in text
        server._tool.search.assert_awaited_once()
        called_params = server._tool.search.await_args.args[0]
        assert called_params.query == "multiple securityfilterchain beans"
        assert called_params.top_k == 3
        assert called_params.module is None
        assert called_params.version is None


@pytest.mark.asyncio
async def test_execute_search_tool_returns_graceful_error() -> None:
    with patch("everspring_mcp.mcp.server.HybridRetriever") as mock_retriever:
        retriever = mock_retriever.return_value
        retriever._embedder.prefetch_model = AsyncMock(return_value=None)
        retriever.ensure_bm25_index.return_value = True

        server = create_server(name="sdk-test")
        server._tool.get_status = AsyncMock(  # type: ignore[method-assign]
            return_value=type("_FakeStatus", (), {"modules": []})()
        )
        server._tool.search = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("Chroma index missing")
        )

        result = await server._execute_search_tool(
            SearchToolArgs(query="multiple securityfilterchain beans"),
        )

        assert result.isError is True
        text = _extract_text(result)
        payload = json.loads(text)
        assert payload["error_type"] == "runtime_unavailable"
        assert "Search is currently unavailable" in payload["message"]
        assert payload["context"]["exception_type"] == "RuntimeError"
        assert "resolution_hints" in payload


@pytest.mark.asyncio
async def test_execute_search_tool_returns_structured_invalid_module_error() -> None:
    with patch("everspring_mcp.mcp.server.HybridRetriever") as mock_retriever:
        retriever = mock_retriever.return_value
        retriever._embedder.prefetch_model = AsyncMock(return_value=None)
        retriever.ensure_bm25_index.return_value = True
        retriever.search = AsyncMock(return_value=[])

        server = create_server(name="sdk-test")
        server._tool.get_status = AsyncMock(  # type: ignore[method-assign]
            return_value=type(
                "_FakeStatus",
                (),
                {
                    "modules": [
                        type(
                            "_Mod", (), {"name": "spring-boot", "versions": [4, 5, 6]}
                        )(),
                        type(
                            "_Mod", (), {"name": "spring-framework", "versions": [6]}
                        )(),
                    ]
                },
            )()
        )

        result = await server._execute_search_tool(
            SearchToolArgs(
                query="security filter chain",
                module="spring-securitty",
            ),
        )

        assert result.isError is True
        payload = json.loads(_extract_text(result))
        assert payload["error_type"] == "invalid_module"
        assert payload["context"]["requested_module"] == "spring-securitty"
        assert "spring-boot" in payload["context"]["available_modules"]
        assert payload["context"]["available_versions_by_module"]["spring-boot"] == [
            4,
            5,
            6,
        ]


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
