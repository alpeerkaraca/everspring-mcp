"""Tests for MCP server, client, and tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from everspring_mcp.mcp.client import MCPClient
from everspring_mcp.mcp.models import (
    ProgressNotification,
    SearchParameters,
    SearchResponse,
    SearchResultItem,
    SearchStatus,
    StatusResponse,
)
from everspring_mcp.mcp.server import create_server
from everspring_mcp.mcp.tools import SpringDocsTool
from everspring_mcp.models.metadata import SearchResult


class TestSearchParameters:
    """Tests for SearchParameters validation."""

    def test_valid_parameters(self):
        """Valid parameters should work."""
        params = SearchParameters(
            query="how to configure DataSource",
            module="spring-boot",
            version=4,
            submodule="jdbc",
            top_k=5,
            score_threshold=0.02,
        )
        assert params.query == "how to configure DataSource"
        assert params.module == "spring-boot"
        assert params.version == 4
        assert params.submodule == "jdbc"
        assert params.top_k == 5
        assert params.score_threshold == 0.02

    def test_minimal_parameters(self):
        """Minimal parameters should use defaults."""
        params = SearchParameters(query="test query")
        assert params.query == "test query"
        assert params.module is None
        assert params.version is None
        assert params.submodule is None
        assert params.top_k == 3
        assert params.score_threshold == 0.01

    def test_query_too_short(self):
        """Query under 3 chars should fail."""
        with pytest.raises(ValueError):
            SearchParameters(query="ab")

    def test_top_k_out_of_range(self):
        """top_k must be 1-10."""
        with pytest.raises(ValueError):
            SearchParameters(query="test query", top_k=0)
        with pytest.raises(ValueError):
            SearchParameters(query="test query", top_k=11)

    def test_score_threshold_bounds(self):
        """score_threshold must be 0.0-1.0."""
        with pytest.raises(ValueError):
            SearchParameters(query="test query", score_threshold=-0.1)
        with pytest.raises(ValueError):
            SearchParameters(query="test query", score_threshold=1.5)


class TestSearchResultItem:
    """Tests for SearchResultItem model."""

    def test_valid_result(self):
        """Valid result should work."""
        item = SearchResultItem(
            rank=1,
            title="SQL Databases",
            url="https://docs.spring.io/spring-boot/reference/data/sql.html",
            module="spring-boot",
            version="v4.0",
            submodule=None,
            score=0.0323,
            score_breakdown="dense: #3, sparse: #1",
            content="Configure a DataSource...",
            has_code=True,
        )
        assert item.rank == 1
        assert item.title == "SQL Databases"
        assert item.score == 0.0323
        assert item.has_code is True

    def test_rank_must_be_positive(self):
        """Rank must be >= 1."""
        with pytest.raises(ValueError):
            SearchResultItem(
                rank=0,
                title="Test",
                url="https://example.com",
                module="spring-boot",
                version="v4.0",
                score=0.01,
                score_breakdown="",
                content="Test",
            )


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_success_response(self):
        """Success response with results."""
        response = SearchResponse(
            status=SearchStatus.SUCCESS,
            message="Found 3 relevant results",
            query="DataSource configuration",
            filters_applied={"module": "spring-boot"},
            results_found=10,
            results_returned=3,
            score_threshold=0.01,
            results=[
                SearchResultItem(
                    rank=1,
                    title="SQL Databases",
                    url="https://example.com",
                    module="spring-boot",
                    version="v4.0",
                    score=0.05,
                    score_breakdown="dense: #1",
                    content="...",
                )
            ],
        )
        assert response.status == SearchStatus.SUCCESS
        assert len(response.results) == 1

    def test_below_threshold_response(self):
        """Response when all results below threshold."""
        response = SearchResponse(
            status=SearchStatus.BELOW_THRESHOLD,
            message="All results below threshold",
            query="rare query",
            results_found=5,
            results_returned=0,
            score_threshold=0.1,
            warnings=["Best score 0.02 is below threshold 0.1"],
        )
        assert response.status == SearchStatus.BELOW_THRESHOLD
        assert len(response.warnings) == 1


class TestProgressNotification:
    """Tests for ProgressNotification model."""

    def test_valid_notification(self):
        """Valid notification should work."""
        notif = ProgressNotification(
            stage="embedding_query",
            message="Processing query...",
            percentage=25.0,
            details={"query_length": 50},
        )
        assert notif.stage == "embedding_query"
        assert notif.percentage == 25.0

    def test_percentage_bounds(self):
        """Percentage must be 0-100."""
        with pytest.raises(ValueError):
            ProgressNotification(
                stage="test",
                message="test",
                percentage=101.0,
            )


class TestSpringDocsTool:
    """Tests for SpringDocsTool."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever."""
        with patch("everspring_mcp.mcp.tools.HybridRetriever") as mock:
            instance = mock.return_value
            instance.ensure_bm25_index.return_value = True
            instance.search = AsyncMock(return_value=[
                SearchResult(
                    id="doc-1",
                    content="Test content about DataSource",
                    title="SQL Databases",
                    url="https://docs.spring.io/test",
                    module="spring-boot",
                    submodule=None,
                    version_major=4,
                    version_minor=0,
                    score=0.05,
                    dense_rank=1,
                    sparse_rank=2,
                    section_path="",
                    has_code=True,
                ),
            ])
            yield instance

    @pytest.fixture
    def mock_chroma(self):
        """Create mock ChromaDB client."""
        with patch("everspring_mcp.mcp.tools.ChromaClient") as mock:
            instance = mock.return_value
            collection = MagicMock()
            collection.count.return_value = 100
            collection.get.return_value = {
                "metadatas": [
                    {"module": "spring-boot", "version_major": 4, "submodule": ""},
                ]
            }
            instance.get_collection.return_value = collection
            yield instance

    @pytest.mark.asyncio
    async def test_search_success(self, mock_retriever, mock_chroma):
        """Successful search returns results."""
        tool = SpringDocsTool()
        await tool.initialize()

        params = SearchParameters(
            query="how to configure DataSource",
            module="spring-boot",
        )

        response = await tool.search(params)

        assert response.status == SearchStatus.SUCCESS
        assert response.results_returned == 1
        assert response.results[0].title == "SQL Databases"
        assert response.results[0].score == 0.05
        assert '<spring_doc_chunk module="spring-boot"' in response.results[0].content
        assert "<breadcrumb>SQL Databases</breadcrumb>" in response.results[0].content
        assert "<content>Test content about DataSource</content>" in response.results[0].content

    @pytest.mark.asyncio
    async def test_search_wraps_and_escapes_xml_chunk_content(self, mock_retriever, mock_chroma):
        """Result content is wrapped in XML with escaped breadcrumb/content."""
        mock_retriever.search.return_value = [
            SearchResult(
                id="doc-1",
                content="Use <bean> & DataSource.",
                title="JDBC Config",
                url="https://docs.spring.io/test",
                module="spring-boot",
                submodule="jdbc",
                version_major=4,
                version_minor=0,
                score=0.05,
                dense_rank=1,
                sparse_rank=2,
                section_path="Data Access > JDBC <Setup>",
                has_code=True,
            ),
        ]

        tool = SpringDocsTool()
        await tool.initialize()

        response = await tool.search(SearchParameters(query="jdbc datasource", top_k=1))
        chunk_xml = response.results[0].content

        assert '<spring_doc_chunk module="spring-boot" version="v4.0"' in chunk_xml
        assert 'submodule="jdbc"' in chunk_xml
        assert (
            "<breadcrumb>Data Access &gt; JDBC &lt;Setup&gt;</breadcrumb>"
            in chunk_xml
        )
        assert "<content>Use &lt;bean&gt; &amp; DataSource.</content>" in chunk_xml
        assert chunk_xml.strip().endswith("</spring_doc_chunk>")

    @pytest.mark.asyncio
    async def test_search_below_threshold(self, mock_retriever, mock_chroma):
        """Results below threshold are filtered."""
        # Make retriever return low-score results
        mock_retriever.search.return_value = [
            SearchResult(
                id="doc-1",
                content="Test",
                title="Test",
                url="https://example.com",
                module="spring-boot",
                submodule=None,
                version_major=4,
                version_minor=0,
                score=0.005,  # Below default threshold
                dense_rank=1,
                sparse_rank=1,
                section_path="",
                has_code=False,
            ),
        ]

        tool = SpringDocsTool()
        await tool.initialize()

        params = SearchParameters(
            query="obscure query",
            score_threshold=0.01,
        )

        response = await tool.search(params)

        assert response.status == SearchStatus.BELOW_THRESHOLD
        assert response.results_returned == 0
        assert len(response.warnings) > 0

    @pytest.mark.asyncio
    async def test_submodule_filter(self, mock_retriever, mock_chroma):
        """Submodule filter is applied."""
        # Return results with different submodules
        mock_retriever.search.return_value = [
            SearchResult(
                id="doc-1",
                content="Redis content",
                title="Redis",
                url="https://example.com/redis",
                module="spring-data",
                submodule="redis",
                version_major=4,
                version_minor=0,
                score=0.05,
                dense_rank=1,
                sparse_rank=1,
                section_path="",
                has_code=False,
            ),
            SearchResult(
                id="doc-2",
                content="JPA content",
                title="JPA",
                url="https://example.com/jpa",
                module="spring-data",
                submodule="jpa",
                version_major=4,
                version_minor=0,
                score=0.04,
                dense_rank=2,
                sparse_rank=2,
                section_path="",
                has_code=False,
            ),
        ]

        tool = SpringDocsTool()
        await tool.initialize()

        params = SearchParameters(
            query="how to configure repository",
            module="spring-data",
            submodule="redis",
        )

        response = await tool.search(params)

        assert response.status == SearchStatus.SUCCESS
        assert response.results_returned == 1
        assert response.results[0].submodule == "redis"

    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_retriever, mock_chroma):
        """Progress callback is invoked."""
        notifications: list[ProgressNotification] = []

        def on_progress(notif: ProgressNotification) -> None:
            notifications.append(notif)

        tool = SpringDocsTool(progress_callback=on_progress)
        await tool.initialize()

        params = SearchParameters(query="test query")
        await tool.search(params)

        assert len(notifications) > 0
        # Should have init, embedding, search, fusion, filtering, complete stages
        stages = {n.stage for n in notifications}
        assert "initializing" in stages or "embedding_query" in stages


class TestMCPServer:
    """Tests for MCPServer."""

    def test_create_server(self):
        """Server can be created."""
        server = create_server(name="test-server")
        assert server.name == "test-server"

    @pytest.mark.asyncio
    async def test_handle_request_search(self):
        """Handle search request."""
        with patch("everspring_mcp.mcp.tools.HybridRetriever") as mock_retriever:
            instance = mock_retriever.return_value
            instance.ensure_bm25_index.return_value = True
            instance.search = AsyncMock(return_value=[])

            with patch("everspring_mcp.mcp.tools.ChromaClient"):
                server = create_server()

                result = await server.handle_request(
                    "search_spring_docs",
                    {"query": "test query"},
                )

                assert "status" in result
                assert "query" in result

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self):
        """Unknown tool raises error."""
        server = create_server()

        with pytest.raises(ValueError, match="Unknown tool"):
            await server.handle_request("unknown_tool", {})


class TestMCPClient:
    """Tests for MCPClient."""

    @pytest.mark.asyncio
    async def test_format_results(self):
        """Format results for display."""
        client = MCPClient(show_progress=False)

        response = SearchResponse(
            status=SearchStatus.SUCCESS,
            message="Found 1 result",
            query="test",
            results_found=1,
            results_returned=1,
            score_threshold=0.01,
            results=[
                SearchResultItem(
                    rank=1,
                    title="Test Doc",
                    url="https://example.com",
                    module="spring-boot",
                    version="v4.0",
                    score=0.05,
                    score_breakdown="dense: #1",
                    content="Test content",
                    has_code=True,
                )
            ],
        )

        formatted = client.format_results(response)

        assert "✓" in formatted  # Success icon
        assert "Test Doc" in formatted
        assert "spring-boot v4.0" in formatted
        assert "Code examples" in formatted

    @pytest.mark.asyncio
    async def test_format_status(self):
        """Format status for display."""
        client = MCPClient(show_progress=False)

        status = StatusResponse(
            healthy=True,
            index_ready=True,
            total_documents=100,
            bm25_index_loaded=True,
        )

        formatted = client.format_status(status)

        assert "✓ Healthy" in formatted
        assert "Ready" in formatted
        assert "100" in formatted


# Integration test placeholder
class TestMCPIntegration:
    """Integration tests for MCP components."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires ChromaDB with indexed data")
    async def test_end_to_end_search(self):
        """Full search flow from client to tool."""
        client = MCPClient()
        await client.initialize()

        response = await client.search(
            query="how to configure DataSource in Spring Boot",
            module="spring-boot",
            version=4,
        )

        assert response.status in (SearchStatus.SUCCESS, SearchStatus.NO_RESULTS)
