"""EverSpring MCP - GitHub data ingestion service.

This module provides GitHubIngester for fetching and processing:
- Release Notes and Migration Guides from GitHub Wikis.
- Project metadata (pom.xml, build.gradle) from repositories.
- Semantic cleaning and section tagging for LLM consumption.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from everspring_mcp.models.base import compute_hash
from everspring_mcp.models.content import ContentType, ScrapedPage
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger("scraper.github")


class GitHubIngestionConfig(BaseModel):
    """Configuration for GitHub ingestion."""
    
    timeout_seconds: int = 30
    user_agent: str = "EverSpring-MCP/1.0"
    github_token: str | None = Field(default=None, description="GitHub API token for higher rate limits")


class GitHubIngester:
    """Service for ingesting documentation from GitHub."""

    def __init__(self, config: GitHubIngestionConfig | None = None):
        self.config = config or GitHubIngestionConfig()
        self.headers = {"User-Agent": self.config.user_agent}
        if self.config.github_token:
            self.headers["Authorization"] = f"token {self.config.github_token}"

    async def fetch_wiki_raw(self, owner: str, repo: str, page_name: str) -> str:
        """Fetch raw markdown content from a GitHub Wiki.
        
        Args:
            owner: Repository owner (e.g., 'spring-projects').
            repo: Repository name (e.g., 'spring-boot').
            page_name: Wiki page name (e.g., 'Spring-Boot-4.0-Release-Notes').
            
        Returns:
            Raw markdown content.
        """
        # GitHub Wiki raw content URL pattern
        url = f"https://raw.githubusercontent.com/wiki/{owner}/{repo}/{page_name}.md"
        
        async with httpx.AsyncClient(headers=self.headers, timeout=self.config.timeout_seconds) as client:
            logger.info(f"Fetching GitHub Wiki: {url}")
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def fetch_repo_file(self, owner: str, repo: str, path: str, branch: str = "main") -> str:
        """Fetch a raw file from a GitHub repository.
        
        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path to the file in the repo.
            branch: Branch name.
            
        Returns:
            Raw file content.
        """
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        
        async with httpx.AsyncClient(headers=self.headers, timeout=self.config.timeout_seconds) as client:
            logger.info(f"Fetching repo file: {url}")
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def clean_documentation_noise(self, content: str) -> str:
        """Clean markdown content by removing boilerplate and navigation noise.
        
        Similar to 'downdoc', this strips:
        - Navigation sidebars/headers
        - 'See also' sections
        - Social media/boilerplate links
        - GitHub-specific wiki metadata
        
        Args:
            content: Raw markdown content.
            
        Returns:
            Cleaned markdown.
        """
        # Remove navigation blocks if they are marked with specific comments
        content = re.sub(r"<!--\s*nav\s*-->.*?<!--\s*/nav\s*-->", "", content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove "See also" sections until the next heading or end of file
        content = re.sub(r"(?i)##?\s+See\s+also.*?(?=\n##?|\Z)", "", content, flags=re.DOTALL)
        
        # Remove common footer noise
        content = re.sub(r"(?i)---\s*\n.*?Back\s+to\s+top.*?\n", "", content)
        
        # Remove GitHub Wiki sidebar links if they appear at the top
        content = re.sub(r"^\[\[.*?\]\]\s*$", "", content, flags=re.MULTILINE)
        
        # Remove repetitive boilerplate links
        content = re.sub(r"\[Edit this page\].*?\n", "", content, flags=re.IGNORECASE)

        # Normalize whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        return content.strip()

    def tag_semantic_sections(self, content: str, content_type: ContentType, version: str) -> str:
        """Wrap logical sections with XML-like tags for LLM context.
        
        Args:
            content: Cleaned markdown content.
            content_type: Type of documentation.
            version: Version string.
            
        Returns:
            Tagged content.
        """
        tag_name = content_type.value.replace("-", "_")
        
        # Split by H2 headers to identify sub-sections
        sections = re.split(r"(?m)^(##\s+.*)$", content)
        
        if len(sections) <= 1:
            return f"<{tag_name} version=\"{version}\">\n{content}\n</{tag_name}>"
        
        tagged_output = [f"<{tag_name} version=\"{version}\">"]
        
        if sections[0].strip():
            tagged_output.append(sections[0].strip())
            
        for i in range(1, len(sections), 2):
            header = sections[i]
            body = sections[i+1] if i+1 < len(sections) else ""
            
            section_title = header.replace("##", "").strip()
            section_slug = section_title.lower().replace(" ", "_").replace("(", "").replace(")", "")
            
            tagged_output.append(f"  <section name=\"{section_slug}\" title=\"{section_title}\">")
            tagged_output.append(f"    {header}")
            tagged_output.append(f"    {body.strip()}")
            tagged_output.append("  </section>")
            
        tagged_output.append(f"</{tag_name}>")
        
        return "\n".join(tagged_output)

    def merge_documentation(
        self, 
        base_content: str, 
        extra_content: list[str],
        base_type: ContentType = ContentType.REFERENCE
    ) -> str:
        """Merge cleaned documentation with existing sources.
        
        Args:
            base_content: The main reference or API doc content.
            extra_content: List of additional documents (release notes, etc.) to merge.
            base_type: The type of the base content.
            
        Returns:
            Merged content string.
        """
        merged = [
            f"<!-- START OF {base_type.value.upper()} -->",
            base_content,
            f"<!-- END OF {base_type.value.upper()} -->",
            "\n"
        ]
        
        for idx, doc in enumerate(extra_content):
            merged.append(f"<!-- SUPPLEMENTAL CONTENT BLOCK {idx+1} -->")
            merged.append(doc)
            merged.append("\n")
            
        return "\n".join(merged)

    async def ingest_spring_framework_metadata(self) -> dict[str, str]:
        """Ingest Spring Framework build metadata (pom.xml, build.gradle).
        
        Returns:
            Dictionary mapping filename to content.
        """
        metadata = {}
        files = ["pom.xml", "build.gradle", "build.gradle.kts"]
        
        for filename in files:
            try:
                content = await self.fetch_repo_file("spring-projects", "spring-framework", filename)
                metadata[filename] = content
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug(f"{filename} not found in spring-framework main branch")
                else:
                    logger.error(f"Error fetching {filename}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error fetching {filename}: {e}")
                
        return metadata

    async def ingest_spring_boot_4_docs(self) -> list[ScrapedPage]:
        """Ingest Spring Boot 4.0 Release Notes and Migration Guide.
        
        Returns:
            List of ScrapedPage objects ready for storage.
        """
        pages = []
        wiki_tasks = [
            ("Spring-Boot-4.0-Release-Notes", ContentType.RELEASE_NOTES, "4.0.0"),
            ("Spring-Boot-4.0-Migration-Guide", ContentType.MIGRATION_GUIDE, "4.0.0"),
        ]
        
        module = SpringModule.SPRING_BOOT
        
        for page_name, ctype, version_str in wiki_tasks:
            try:
                raw_md = await self.fetch_wiki_raw("spring-projects", "spring-boot", page_name)
                cleaned_md = self.clean_documentation_noise(raw_md)
                tagged_md = self.tag_semantic_sections(cleaned_md, ctype, version_str)
                
                version = SpringVersion.parse(f"spring-boot:{version_str}")
                url = f"https://github.com/spring-projects/spring-boot/wiki/{page_name}"
                
                page = ScrapedPage.create(
                    url=url,
                    module=module,
                    version=version,
                    submodule=None,
                    title=page_name.replace("-", " "),
                    raw_html=f"<html><body><pre>{raw_md}</pre></body></html>",
                    markdown_content=tagged_md,
                    content_type=ctype
                )
                pages.append(page)
                
            except Exception as e:
                logger.error(f"Failed to ingest {page_name}: {e}")
                
        return pages
