"""Tests for GitHubIngester."""

from unittest.mock import AsyncMock, patch

import pytest

from everspring_mcp.scraper.github_ingester import GitHubIngester
from everspring_mcp.models.content import ContentType


@pytest.mark.asyncio
async def test_fetch_wiki_raw():
    ingester = GitHubIngester()
    owner, repo, page = "spring-projects", "spring-boot", "Spring-Boot-4.0-Release-Notes"
    
    mock_response = AsyncMock()
    mock_response.text = "# Release Notes\n\nSome content."
    mock_response.status_code = 200
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        content = await ingester.fetch_wiki_raw(owner, repo, page)
        assert content == "# Release Notes\n\nSome content."


def test_clean_documentation_noise():
    ingester = GitHubIngester()
    dirty_md = """
[[Sidebar Link]]
# Title
Content.
## See also
* [Link](http://example.com)
---
Back to top
"""
    cleaned = ingester.clean_documentation_noise(dirty_md)
    assert "[[Sidebar Link]]" not in cleaned
    assert "See also" not in cleaned
    assert "Back to top" not in cleaned
    assert "# Title" in cleaned


def test_tag_semantic_sections():
    ingester = GitHubIngester()
    md = """
# Intro
Some intro.
## Section 1
Content 1.
## Section 2
Content 2.
"""
    tagged = ingester.tag_semantic_sections(md, ContentType.RELEASE_NOTES, "4.0.0")
    
    assert "<release_notes version=\"4.0.0\">" in tagged
    assert "</release_notes>" in tagged
    assert "<section name=\"section_1\" title=\"Section 1\">" in tagged
    assert "## Section 1" in tagged
    assert "Content 1." in tagged


def test_merge_documentation():
    ingester = GitHubIngester()
    base = "Base doc content"
    extra = ["<release_notes>Extra 1</release_notes>", "<migration_guide>Extra 2</migration_guide>"]
    
    merged = ingester.merge_documentation(base, extra)
    
    assert "START OF REFERENCE" in merged
    assert "Base doc content" in merged
    assert "SUPPLEMENTAL CONTENT BLOCK 1" in merged
    assert "Extra 1" in merged
    assert "Extra 2" in merged
