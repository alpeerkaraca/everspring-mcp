"""Tests for sync orchestrator document persistence behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.orchestrator import SyncOrchestrator
from everspring_mcp.sync.s3_sync import S3SyncService, SnapshotDownloadResult


@pytest.fixture
def orchestrator_config(tmp_path: Path) -> SyncConfig:
    """Create sync config rooted in a temporary local data directory."""
    config = SyncConfig(
        s3_bucket="test-bucket",
        s3_region="us-east-1",
        s3_prefix="test-docs",
        local_data_dir=tmp_path,
    )
    config.ensure_directories()
    return config


@pytest.mark.asyncio
async def test_sync_module_persists_docs_relative_path_and_submodule(
    mock_s3: Any,
    orchestrator_config: SyncConfig,
) -> None:
    """Synced records should keep docs-relative file path and resolved submodule."""
    doc_key = "test-docs/raw-data/spring-cloud-netflix/5.0.1/abc123/document.md"
    meta_key = "test-docs/raw-data/spring-cloud-netflix/5.0.1/abc123/metadata.json"

    mock_s3.put_object(
        Bucket=orchestrator_config.s3_bucket,
        Key=doc_key,
        Body=b"# Spring Cloud Netflix\n\nReference content.",
        ContentType="text/markdown",
    )
    mock_s3.put_object(
        Bucket=orchestrator_config.s3_bucket,
        Key=meta_key,
        Body=json.dumps(
            {
                "url": "https://docs.spring.io/spring-cloud-netflix/reference/",
                "title": "Spring Cloud Netflix",
                "submodule": "netflix",
                "scraped_at": "2026-04-06T15:00:00Z",
            }
        ).encode("utf-8"),
        ContentType="application/json",
    )

    s3_service = S3SyncService(orchestrator_config, s3_client=mock_s3)
    async with SyncOrchestrator(
        config=orchestrator_config,
        s3_service=s3_service,
    ) as orchestrator:
        result = await orchestrator.sync_module(
            module="spring-cloud",
            version="5.0.1",
            submodule="netflix",
            force=True,
        )

        assert result.status.value == "completed"
        doc = await orchestrator.storage.documents.get_by_s3_key(doc_key)
        assert doc is not None
        assert doc.file_path == "spring-cloud-netflix/5.0.1/abc123/document.md"
        assert doc.submodule == "netflix"

    manifest_key = "test-docs/raw-data/spring-cloud-netflix/5.0.1/manifest.json"
    uploaded_manifest = mock_s3.get_object(
        Bucket=orchestrator_config.s3_bucket,
        Key=manifest_key,
    )
    payload = json.loads(uploaded_manifest["Body"].read().decode("utf-8"))
    assert payload["file_count"] == 2


@pytest.mark.asyncio
async def test_download_latest_snapshots_closes_and_reopens_storage(
    orchestrator_config: SyncConfig,
) -> None:
    """snapshot-download should temporarily close SQLite and reconnect after apply."""

    class FakeSnapshotService:
        async def download_latest_db_snapshots(
            self,
            cleanup_local_archives: bool = True,
            model_name: str | None = None,
            tier: str | None = None,
        ) -> SnapshotDownloadResult:
            del cleanup_local_archives, model_name, tier
            return SnapshotDownloadResult(
                snapshot_token="2026_04_08",
                chroma_snapshot="chroma_db_2026_04_08.zip",
                sqlite_snapshot="sqlite_metadata_2026_04_08.zip",
                bytes_downloaded=2048,
                success=True,
            )

    async with SyncOrchestrator(
        config=orchestrator_config,
        s3_service=FakeSnapshotService(),  # type: ignore[arg-type]
    ) as orchestrator:
        assert orchestrator.storage.db is not None
        result = await orchestrator.download_latest_snapshots()
        assert result.success
        assert orchestrator.storage.db is not None
