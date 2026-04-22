"""EverSpring MCP - GitHub data ingestion service.

This module provides GitHubIngester for fetching and processing:
- Release Notes, Migration Guides, and Changelogs via local Wiki clones.
- Conversion of AsciiDoc to Markdown using 'downdoc'.
- Project metadata (pom.xml, build.gradle) from repositories.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from everspring_mcp.models.base import compute_hash
from everspring_mcp.models.content import ContentType, ScrapedPage
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.utils.logging import get_logger

if TYPE_CHECKING:
    from everspring_mcp.scraper.pipeline import S3Client

logger = get_logger("scraper.github")


class GitHubIngestionConfig(BaseModel):
    """Configuration for GitHub ingestion."""
    
    timeout_seconds: int = 60
    user_agent: str = "EverSpring-MCP/1.0"
    github_token: str | None = Field(default=None, description="GitHub API token")
    downdoc_command: str = "npx downdoc"
    git_command: str = "git"


class GitHubIngester:
    """Service for ingesting documentation from GitHub via local processing."""

    def __init__(self, config: GitHubIngestionConfig | None = None):
        self.config = config or GitHubIngestionConfig()
        self.headers = {"User-Agent": self.config.user_agent}
        if self.config.github_token:
            self.headers["Authorization"] = f"token {self.config.github_token}"

    async def fetch_repo_file(self, owner: str, repo: str, path: str, branch: str = "main") -> str:
        """Fetch a raw file from a GitHub repository (used for pom.xml/build.gradle)."""
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        
        async with httpx.AsyncClient(headers=self.headers, timeout=self.config.timeout_seconds) as client:
            logger.info(f"Fetching repo file: {url}")
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def clone_wiki(self, owner: str, repo: str, dest_dir: Path) -> None:
        """Clone a GitHub Wiki repository locally."""
        wiki_url = f"https://github.com/{owner}/{repo}.wiki.git"
        logger.info(f"Cloning wiki: {wiki_url} to {dest_dir}")
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"

        try:
            subprocess.run(
                [self.config.git_command, "clone", "--depth", "1", wiki_url, str(dest_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone wiki {wiki_url}: {e.stderr}")
            raise

    def find_latest_wiki_files(self, wiki_dir: Path) -> list[Path]:
        """Find the latest 2 minor versions of target docs recursively, excluding preview folders."""
        target_patterns = ["Release-Notes", "Migration-Guide", "Configuration-Changelog"]
        
        # Structure: category -> minor_version (e.g. "4.1") -> latest_file_info
        grouped_files: dict[str, dict[str, dict[str, Any]]] = {p: {} for p in target_patterns}

        all_files = list(wiki_dir.rglob("*.asciidoc")) + list(wiki_dir.rglob("*.adoc"))
        version_regex = re.compile(r"(\d+\.\d+)(?:\.\d+)?")

        for file_path in all_files:
            if "preview" in [part.lower() for part in file_path.parts]:
                continue

            name = file_path.name
            for pattern in target_patterns:
                if pattern in name:
                    match = version_regex.search(name)
                    if match:
                        minor_ver = match.group(1)
                        full_ver = match.group(0)
                        
                        existing = grouped_files[pattern].get(minor_ver)
                        if not existing or self._is_newer(full_ver, existing["version"]):
                            grouped_files[pattern][minor_ver] = {
                                "path": file_path,
                                "version": full_ver,
                                "minor": minor_ver
                            }

        result_paths = []
        for pattern in target_patterns:
            # Get all minor versions found for this pattern
            minors = list(grouped_files[pattern].values())
            
            # Sort by version descending
            def version_key(x):
                parts = x["minor"].split(".")
                return [int(p) if p.isdigit() else p for p in parts]
            
            minors.sort(key=version_key, reverse=True)
            
            # Take top 2 latest minor version documents
            for item in minors[:2]:
                result_paths.append(item["path"])
        
        return result_paths

    def _is_newer(self, v1: str, v2: str) -> bool:
        """Simple semantic version comparison."""
        def parse(v):
            return [int(x) if x.isdigit() else x for x in v.split(".")]
        try:
            return parse(v1) > parse(v2)
        except (ValueError, TypeError, IndexError):
            return v1 > v2

    def _map_pattern_to_type(self, pattern: str) -> ContentType:
        """Map filename patterns to ContentType."""
        if "Release-Notes" in pattern:
            return ContentType.RELEASE_NOTES
        if "Migration-Guide" in pattern:
            return ContentType.MIGRATION_GUIDE
        if "Configuration-Changelog" in pattern:
            return ContentType.REFERENCE 
        return ContentType.REFERENCE

    def convert_to_markdown(self, adoc_path: Path) -> str:
        """Convert AsciiDoc to Markdown using 'downdoc -o'."""
        output_md = adoc_path.with_suffix(".md")
        logger.info(f"Converting {adoc_path} to {output_md}")
        
        try:
            import sys
            import shlex
            
            # Resolve the base command
            base_cmd = shlex.split(self.config.downdoc_command)
            
            use_shell = sys.platform == "win32"
            
            # If using shell=True on Windows, it's often safer to pass the
            # command as a single string rather than a list.
            if use_shell:
                cmd_str = f'{" ".join(base_cmd)} "{adoc_path}" -o "{output_md}"'
                subprocess.run(
                    cmd_str,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    stdin=subprocess.DEVNULL,
                    shell=True
                )
            else:
                # POSIX (Linux/Mac/Fargate) execution
                cmd = base_cmd + [str(adoc_path), "-o", str(output_md)]
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    stdin=subprocess.DEVNULL,
                    shell=False
                )
            
            if not output_md.exists():
                raise FileNotFoundError(f"downdoc failed to create {output_md}")
                
            content = output_md.read_text(encoding="utf-8")
            return content
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {adoc_path}: {e.stderr}")
            raise
        finally:
            # Cleanup generated file if it exists
            if output_md.exists():
                output_md.unlink()

    def clean_documentation_noise(self, content: str) -> str:
        """Clean markdown content by removing boilerplate and navigation noise."""
        content = re.sub(r"<!--\s*nav\s*-->.*?<!--\s*/nav\s*-->", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"(?i)##?\s+See\s+also.*?(?=\n##?|\Z)", "", content, flags=re.DOTALL)
        content = re.sub(r"(?i)---\s*\n.*?Back\s+to\s+top.*?\n", "", content)
        content = re.sub(r"^\[\[.*?\]\]\s*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"\[Edit this page\].*?\n", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def tag_semantic_sections(self, content: str, content_type: ContentType, version: str) -> str:
        """Wrap logical sections with XML-like tags for LLM context."""
        tag_name = content_type.value.replace("-", "_")
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

    async def wiki_exists(self, owner: str, repo: str) -> bool:
        """Check if a GitHub Wiki exists for the given repository."""
        wiki_url = f"https://github.com/{owner}/{repo}.wiki.git"
        async with httpx.AsyncClient(headers=self.headers, timeout=10) as client:
            try:
                # We check the main wiki URL. GitHub Wikis usually return 200/302 if they exist.
                # Git URLs themselves might not be directly browsable via HTTP in the same way,
                # but the base wiki page usually is.
                web_url = f"https://github.com/{owner}/{repo}/wiki"
                response = await client.head(web_url, follow_redirects=True)
                return response.status_code == 200
            except Exception as e:
                logger.debug(f"Error checking wiki existence: {e}")
                return False

    async def ingest_wiki(self, owner: str, repo: str, module: SpringModule | None = None, s3_client: S3Client | None = None) -> list[ScrapedPage]:
        """Orchestrate the full wiki ingestion workflow."""
        if not await self.wiki_exists(owner, repo):
            logger.warning(f"Skipping. Wiki doesn't exist for {owner}/{repo}")
            return []

        pages = []
        module = module or SpringModule.GITHUB_WIKI
        with tempfile.TemporaryDirectory() as tmp_dir:
            wiki_dir = Path(tmp_dir) / "wiki"
            
            # 1. Clone
            self.clone_wiki(owner, repo, wiki_dir)
            # DEBUG
            # 2. Find latest files
            target_files = self.find_latest_wiki_files(wiki_dir)
            
            version_regex = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
            for adoc_path in target_files:
                try:
                    # 3. Convert
                    raw_md = self.convert_to_markdown(adoc_path)
                    # 4. Clean and Tag
                    cleaned_md = self.clean_documentation_noise(raw_md)

                    
                    version_match = version_regex.search(adoc_path.name)
                    version_str = version_match.group(1) if version_match else "unknown"
                    
                    ctype = self._map_pattern_to_type(adoc_path.name)
                    tagged_md = self.tag_semantic_sections(cleaned_md, ctype, version_str)

                    
                    # 5. Create ScrapedPage
                    version = SpringVersion.parse(module, version_str)
                    url = f"https://github.com/{owner}/{repo}/wiki/{adoc_path.stem}"
                    
                    page = ScrapedPage.create(
                        url=url,
                        module=module,
                        version=version,
                        submodule=None,
                        title=adoc_path.stem.replace("-", " "),
                        raw_html=f"<html><body><pre>{raw_md}</pre></body></html>",
                        markdown_content=tagged_md,
                        content_type=ctype
                    )
                    
                    # 6. Optional S3 Upload
                    if s3_client:
                        url_hash = compute_hash(url)[:16]
                        s3_key = f"{module.value}/{version.version_string}/{url_hash}/document.md"
                        s3_client.upload_content(
                            content=tagged_md,
                            key=s3_key,
                            content_hash=compute_hash(tagged_md),
                            metadata={
                                "source-url": url,
                                "module": module.value,
                                "version": version.version_string,
                                "title": page.title,
                                "content-type": ctype.value
                            }
                        )
                    
                    pages.append(page)
                    
                except Exception as e:
                    logger.error(f"Failed to process {adoc_path.name}: {e}")
        
        return pages
