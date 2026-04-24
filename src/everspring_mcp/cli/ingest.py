from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from everspring_mcp.models.content import ContentType, ScrapedPage
from everspring_mcp.models.base import compute_hash
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.github_ingester import GitHubIngester, GitHubIngestionConfig
from everspring_mcp.scraper.pipeline import S3Client, PipelineConfig
from everspring_mcp.storage.repository import StorageManager
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.cli.utils import _parse_module

logger = logging.getLogger("everspring_mcp")

async def _run_ingest_github(args: argparse.Namespace) -> int:
    ingest_config = GitHubIngestionConfig(
        github_token=args.token or os.environ.get("GITHUB_TOKEN")
    )
    ingester = GitHubIngester(ingest_config)

    s3_client = None
    if args.s3_bucket or args.s3_region or args.s3_prefix:
        s3_client = S3Client(
            bucket=args.s3_bucket or "everspring-mcp-kb",
            prefix=args.s3_prefix or "spring-docs/raw-data",
            region=args.s3_region or "eu-central-1",
        )
    else:
        try:
            pipeline_config = PipelineConfig.from_env()
            s3_client = S3Client(
                bucket=pipeline_config.s3_bucket,
                prefix=pipeline_config.s3_prefix,
                region=pipeline_config.aws_region,
            )
        except Exception as e:
            logger.debug(f"S3 not configured via env, skipping upload: {e}")

    db_path = Path(args.data_dir) / "everspring.db" if args.data_dir else VectorConfig.from_env().db_path
    storage = StorageManager(db_path)
    await storage.connect()

    results = []
    module = args.module or SpringModule.GITHUB_WIKI
    version_str = args.version or "0.0.0"
    
    try:
        if args.repo:
            logger.info(f"Starting Wiki ingestion for {args.owner}/{args.repo}")
            pages = await ingester.ingest_wiki(
                owner=args.owner,
                repo=args.repo,
                module=module,
                s3_client=s3_client
            )
            for page in pages:
                await storage.documents.save(page)
                
                results.append({
                    "type": "wiki",
                    "name": page.title,
                    "status": "success",
                    "version": page.version.version_string
                })

        if args.files:
            for file_path in args.files:
                try:
                    logger.info(f"Ingesting Repo file: {file_path}")
                    content = await ingester.fetch_repo_file(args.owner, args.repo, file_path)
                    
                    version = SpringVersion.parse(module, version_str)
                    url = f"https://github.com/{args.owner}/{args.repo}/blob/main/{file_path}"
                    
                    page = ScrapedPage.create(
                        url=url,
                        module=module,
                        version=version,
                        submodule=None,
                        title=file_path,
                        raw_html=f"<html><body><pre>{content}</pre></body></html>",
                        markdown_content=content,
                        content_type=ContentType.REFERENCE
                    )
                    
                    await storage.documents.save(page)

                    if s3_client:
                        url_hash = compute_hash(url)[:16]
                        s3_key = f"{module.value}/{version.version_string}/metadata/{file_path}"
                        
                        s3_client.upload_content(
                            content=content,
                            key=s3_key,
                            content_hash=compute_hash(content),
                            metadata={
                                "source-url": url,
                                "module": module.value,
                                "file-path": file_path
                            }
                        )
                        results.append({"type": "file", "name": file_path, "status": "success", "s3_key": s3_key})
                    else:
                        results.append({"type": "file", "name": file_path, "status": "fetched", "content_length": len(content)})
                        
                except Exception as e:
                    logger.error(f"Failed to ingest repo file {file_path}: {e}")
                    results.append({"type": "file", "name": file_path, "status": "failed", "error": str(e)})

    finally:
        await storage.close()

    if args.json:
        logger.info(json.dumps(results, indent=2))
    else:
        success = sum(1 for r in results if r["status"] == "success" or r["status"] == "fetched")
        logger.info(f"Ingestion complete: {success} succeeded, {len(results) - success} failed")
        
    return 0 if all(r["status"] != "failed" for r in results) else 1


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    github = subparsers.add_parser(
        "ingest-github",
        help="Ingest docs from GitHub Wiki or Repo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    github.add_argument(
        "--owner", default="spring-projects", help="GitHub repository owner"
    )
    github.add_argument(
        "--repo", required=True, help="GitHub repository name (e.g., spring-boot)"
    )
    github.add_argument(
        "--pages",
        nargs="+",
        help="Wiki page names to ingest (for Release Notes/Migration Guides)",
    )
    github.add_argument(
        "--files",
        nargs="+",
        help="Repository file paths to ingest (e.g., pom.xml, build.gradle)",
    )
    github.add_argument(
        "--module", default=None, type=_parse_module, help="Spring module"
    )
    github.add_argument("--version", default=None, help="Version string (e.g., 4.0.0)")
    github.add_argument("--token", default=None, help="GitHub API token")
    github.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    github.add_argument("--s3-region", default=None, help="S3 region override")
    github.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    github.add_argument("--data-dir", default=None, help="Local data directory override")
    github.add_argument("--json", action="store_true", help="Output JSON summary")
    github.set_defaults(func=_run_ingest_github)
