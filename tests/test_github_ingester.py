"""Tests for GitHubIngester - Local Processing Workflow."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from everspring_mcp.scraper.github_ingester import GitHubIngester, GitHubIngestionConfig
from everspring_mcp.models.spring import SpringModule


@pytest.fixture
def ingester():
    return GitHubIngester()


def test_find_latest_wiki_files(ingester, tmp_path):
    # Create fake asciidoc files with versions
    (tmp_path / "Spring-Boot-4.0-Release-Notes.asciidoc").write_text("v4.0")
    (tmp_path / "Spring-Boot-4.1-Release-Notes.asciidoc").write_text("v4.1")
    (tmp_path / "Spring-Boot-4.0-Migration-Guide.asciidoc").write_text("v4.0 migration")
    (tmp_path / "Spring-Boot-4.0-Configuration-Changelog.asciidoc").write_text("v4.0 changelog")
    (tmp_path / "Random-File.txt").write_text("random")

    latest_files = ingester.find_latest_wiki_files(tmp_path)
    
    file_names = [f.name for f in latest_files]
    assert len(file_names) == 3
    assert "Spring-Boot-4.1-Release-Notes.asciidoc" in file_names
    assert "Spring-Boot-4.0-Migration-Guide.asciidoc" in file_names
    assert "Spring-Boot-4.0-Configuration-Changelog.asciidoc" in file_names
    assert "Spring-Boot-4.0-Release-Notes.asciidoc" not in file_names


@patch("subprocess.run")
def test_clone_wiki(mock_run, ingester, tmp_path):
    dest = tmp_path / "wiki"
    ingester.clone_wiki("spring-projects", "spring-boot", dest)
    
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "clone" in args
    assert "https://github.com/spring-projects/spring-boot.wiki.git" in args


@patch("subprocess.run")
def test_convert_to_markdown(mock_run, ingester, tmp_path):
    mock_run.return_value = MagicMock(stdout="# Converted Markdown", check=True)
    adoc = tmp_path / "test.asciidoc"
    adoc.write_text("= Test")
    
    md = ingester.convert_to_markdown(adoc)
    assert md == "# Converted Markdown"
    mock_run.assert_called_once()


@pytest.mark.asyncio
@patch.object(GitHubIngester, "clone_wiki")
@patch.object(GitHubIngester, "find_latest_wiki_files")
@patch.object(GitHubIngester, "convert_to_markdown")
async def test_ingest_wiki_orchestration(mock_convert, mock_find, mock_clone, ingester, tmp_path):
    mock_find.return_value = [Path("Spring-Boot-4.1-Release-Notes.asciidoc")]
    mock_convert.return_value = "## Converted Content"
    
    pages = await ingester.ingest_wiki("spring-projects", "spring-boot", SpringModule.BOOT)
    
    assert len(pages) == 1
    assert pages[0].version.version_string == "4.1.0"
    assert "Converted Content" in pages[0].markdown_content
    mock_clone.assert_called_once()
