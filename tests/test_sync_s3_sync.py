"""Tests for separated DB snapshot uploads to S3."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from everspring_mcp.models.base import compute_hash
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.s3_sync import S3SyncService


@pytest.fixture
def sync_config(tmp_path: Path) -> SyncConfig:
    """Create sync config rooted in a temporary local data directory."""
    config = SyncConfig(
        s3_bucket="test-bucket",
        s3_region="us-east-1",
        s3_prefix="test-docs",
        local_data_dir=tmp_path,
    )
    config.ensure_directories()
    return config


def _seed_local_datastores(config: SyncConfig) -> tuple[bytes, bytes]:
    """Write sample SQLite and ChromaDB content."""
    db_bytes = b"sqlite-local-content"
    chroma_bytes = b"vector-local-content"

    config.db_path.write_bytes(db_bytes)
    chroma_file = config.chroma_dir / "spring_docs" / "index.bin"
    chroma_file.parent.mkdir(parents=True, exist_ok=True)
    chroma_file.write_bytes(chroma_bytes)

    return db_bytes, chroma_bytes


@pytest.mark.asyncio
async def test_upload_db_snapshots_uses_separated_snapshot_keys(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Snapshots should upload to db-snapshots with required date-based names."""
    expected_db, expected_chroma = _seed_local_datastores(sync_config)
    service = S3SyncService(sync_config, s3_client=mock_s3)
    snapshot_date = date(2026, 4, 6)

    results = await service.upload_db_snapshots(snapshot_date=snapshot_date)

    assert len(results) == 2
    assert all(r.success for r in results)

    expected_chroma_key = "test-docs/db-snapshots/chroma_db_2026_04_06.zip"
    expected_sqlite_key = "test-docs/db-snapshots/sqlite_metadata_2026_04_06.zip"
    by_key = {r.s3_key: r for r in results}
    assert set(by_key) == {expected_chroma_key, expected_sqlite_key}

    chroma_obj = mock_s3.get_object(Bucket=sync_config.s3_bucket, Key=expected_chroma_key)
    chroma_payload = chroma_obj["Body"].read()
    assert chroma_obj["Metadata"]["snapshot-type"] == "chroma_db"
    assert chroma_obj["Metadata"]["content-hash"] == compute_hash(chroma_payload)
    with zipfile.ZipFile(io.BytesIO(chroma_payload), mode="r") as archive:
        assert archive.read("spring_docs/index.bin") == expected_chroma

    sqlite_obj = mock_s3.get_object(Bucket=sync_config.s3_bucket, Key=expected_sqlite_key)
    sqlite_payload = sqlite_obj["Body"].read()
    assert sqlite_obj["Metadata"]["snapshot-type"] == "sqlite_metadata"
    assert sqlite_obj["Metadata"]["content-hash"] == compute_hash(sqlite_payload)
    with zipfile.ZipFile(io.BytesIO(sqlite_payload), mode="r") as archive:
        assert archive.read(sync_config.db_filename) == expected_db


@pytest.mark.asyncio
async def test_upload_db_snapshots_cleans_local_archives_when_requested(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Local snapshot zip files should be removed when cleanup flag is enabled."""
    _seed_local_datastores(sync_config)
    service = S3SyncService(sync_config, s3_client=mock_s3)
    snapshot_date = date(2026, 4, 6)

    results = await service.upload_db_snapshots(
        snapshot_date=snapshot_date,
        cleanup_local_archives=True,
    )

    assert all(r.success for r in results)
    assert not sync_config.get_chroma_snapshot_local_path(snapshot_date).exists()
    assert not sync_config.get_sqlite_snapshot_local_path(snapshot_date).exists()


@pytest.mark.asyncio
async def test_upload_db_snapshots_reports_failure_when_sqlite_missing(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Missing SQLite DB should produce a failed sqlite snapshot result."""
    chroma_file = sync_config.chroma_dir / "spring_docs" / "index.bin"
    chroma_file.parent.mkdir(parents=True, exist_ok=True)
    chroma_file.write_bytes(b"vector-local-content")
    if sync_config.db_path.exists():
        sync_config.db_path.unlink()

    service = S3SyncService(sync_config, s3_client=mock_s3)
    results = await service.upload_db_snapshots(snapshot_date=date(2026, 4, 6))

    assert len(results) == 2
    failed = [r for r in results if not r.success]
    assert len(failed) == 1
    assert failed[0].snapshot_name == "sqlite_metadata_2026_04_06.zip"
    assert failed[0].error is not None
    assert "sqlite database not found" in failed[0].error.lower()
