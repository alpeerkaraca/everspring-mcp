"""Tests for separated DB snapshot uploads to S3."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from everspring_mcp.models.base import compute_hash
from everspring_mcp.models.spring import SpringModule
from everspring_mcp.models.sync import ChangeType, FileChange, SyncDelta
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.s3_sync import DownloadResult, S3SyncService


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


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    """Build an in-memory zip payload from path->bytes entries."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, payload in entries.items():
            archive.writestr(path, payload)
    return buffer.getvalue()


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

    expected_chroma_key = "test-docs/db-snapshots/bge-m3-main/chroma_db_2026_04_06.zip"
    expected_sqlite_key = "test-docs/db-snapshots/bge-m3-main/sqlite_metadata_2026_04_06.zip"
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


@pytest.mark.asyncio
async def test_upload_db_snapshots_reports_failure_when_chroma_empty(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Empty ChromaDB directory should fail instead of uploading placeholder archive."""
    sync_config.db_path.write_bytes(b"sqlite-local-content")
    sync_config.chroma_dir.mkdir(parents=True, exist_ok=True)

    service = S3SyncService(sync_config, s3_client=mock_s3)
    results = await service.upload_db_snapshots(snapshot_date=date(2026, 4, 6))

    assert len(results) == 2
    failed = [r for r in results if not r.success]
    assert len(failed) == 1
    assert failed[0].snapshot_name == "chroma_db_2026_04_06.zip"
    assert failed[0].error is not None
    assert "chroma" in failed[0].error.lower()
    assert "empty" in failed[0].error.lower()


@pytest.mark.asyncio
async def test_upload_db_snapshots_includes_bm25_when_present(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """SQLite snapshot archive should include bm25_index.pkl when available locally."""
    expected_db, _ = _seed_local_datastores(sync_config)
    expected_bm25 = b"bm25-index-content"
    bm25_path = sync_config.local_data_dir / "bm25_index.pkl"
    bm25_path.write_bytes(expected_bm25)

    service = S3SyncService(sync_config, s3_client=mock_s3)
    snapshot_date = date(2026, 4, 6)
    await service.upload_db_snapshots(snapshot_date=snapshot_date)

    sqlite_key = "test-docs/db-snapshots/bge-m3-main/sqlite_metadata_2026_04_06.zip"
    sqlite_obj = mock_s3.get_object(Bucket=sync_config.s3_bucket, Key=sqlite_key)
    sqlite_payload = sqlite_obj["Body"].read()
    with zipfile.ZipFile(io.BytesIO(sqlite_payload), mode="r") as archive:
        assert archive.read(sync_config.db_filename) == expected_db
        assert archive.read("bm25_index.pkl") == expected_bm25


@pytest.mark.asyncio
async def test_build_manifest_from_s3_reads_separated_raw_data_prefix(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Manifest fallback should read files from separated raw-data module-submodule prefix."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    base_prefix = sync_config.get_raw_data_prefix(
        module="spring-data",
        version="5.0.4",
        submodule="jpa",
    )
    doc_key = f"{base_prefix}/abc123/document.md"
    meta_key = f"{base_prefix}/abc123/metadata.json"

    mock_s3.put_object(
        Bucket=sync_config.s3_bucket,
        Key=doc_key,
        Body=b"# Spring Data JPA",
        ContentType="text/markdown",
    )
    mock_s3.put_object(
        Bucket=sync_config.s3_bucket,
        Key=meta_key,
        Body=b'{"title":"Spring Data JPA"}',
        ContentType="application/json",
    )

    manifest = await service.build_manifest_from_s3(
        module="spring-data",
        version="5.0.4",
        submodule="jpa",
    )

    assert manifest.file_count == 2
    assert sorted(entry.path for entry in manifest.files) == [
        "abc123/document.md",
        "abc123/metadata.json",
    ]


@pytest.mark.asyncio
async def test_build_manifest_from_s3_supports_legacy_prefix(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Manifest fallback should remain compatible with legacy module/submodule paths."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    legacy_key = f"{sync_config.s3_prefix}/spring-data/jpa/5.0.4/legacy/document.md"

    mock_s3.put_object(
        Bucket=sync_config.s3_bucket,
        Key=legacy_key,
        Body=b"# Legacy path document",
        ContentType="text/markdown",
    )

    manifest = await service.build_manifest_from_s3(
        module="spring-data",
        version="5.0.4",
        submodule="jpa",
    )

    assert manifest.file_count == 1
    assert manifest.files[0].path == "legacy/document.md"


@pytest.mark.asyncio
async def test_upload_manifest_writes_manifest_json_to_expected_key(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Uploaded manifest should be stored under raw-data module-submodule/version path."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    manifest = await service.build_manifest_from_s3(
        module="spring-data",
        version="5.0.4",
        submodule="jpa",
    )

    uploaded_key = await service.upload_manifest(
        module="spring-data",
        version="5.0.4",
        manifest=manifest,
        submodule="jpa",
    )

    assert uploaded_key == "test-docs/raw-data/spring-data-jpa/5.0.4/manifest.json"
    obj = mock_s3.get_object(Bucket=sync_config.s3_bucket, Key=uploaded_key)
    body = obj["Body"].read().decode("utf-8")
    payload = json.loads(body)
    assert payload["version"] == "5.0.4"
    assert payload["file_count"] == 0
    assert obj["Metadata"]["manifest-version"] == "5.0.4"
    assert "content-hash" in obj["Metadata"]


@pytest.mark.asyncio
async def test_ensure_manifest_uploads_when_missing(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """ensure_manifest should build/upload when manifest file does not exist yet."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_raw_data_prefix("spring-data", "5.0.4", submodule="jpa")

    mock_s3.put_object(
        Bucket=sync_config.s3_bucket,
        Key=f"{prefix}/abc123/document.md",
        Body=b"# Spring Data JPA",
        ContentType="text/markdown",
        Metadata={"content-hash": compute_hash("# Spring Data JPA")},
    )

    result = await service.ensure_manifest(
        module="spring-data",
        version="5.0.4",
        submodule="jpa",
    )

    assert result.status == "uploaded"
    assert result.file_count == 1
    assert result.manifest_key == "test-docs/raw-data/spring-data-jpa/5.0.4/manifest.json"
    manifest_obj = mock_s3.get_object(Bucket=sync_config.s3_bucket, Key=result.manifest_key)
    manifest_payload = json.loads(manifest_obj["Body"].read().decode("utf-8"))
    assert manifest_payload["file_count"] == 1


@pytest.mark.asyncio
async def test_ensure_manifest_skips_when_existing_and_not_forced(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """ensure_manifest should skip rebuild when manifest already exists unless force is set."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    empty_manifest = await service.build_manifest_from_s3("spring-data", "5.0.4", submodule="jpa")
    await service.upload_manifest("spring-data", "5.0.4", empty_manifest, submodule="jpa")

    result = await service.ensure_manifest(
        module="spring-data",
        version="5.0.4",
        submodule="jpa",
        force=False,
    )

    assert result.status == "skipped"
    assert result.manifest_key == "test-docs/raw-data/spring-data-jpa/5.0.4/manifest.json"


def test_s3_service_uses_parallel_jobs_for_boto_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
    sync_config: SyncConfig,
) -> None:
    """S3 client should be configured with max_pool_connections from parallel_jobs."""
    captured: dict[str, object] = {}

    class DummyS3Client:
        pass

    def fake_boto_client(service_name: str, **kwargs: object) -> DummyS3Client:
        captured["service_name"] = service_name
        captured["kwargs"] = kwargs
        return DummyS3Client()

    monkeypatch.setattr("everspring_mcp.sync.s3_sync.boto3.client", fake_boto_client)
    config = sync_config.model_copy(update={"parallel_jobs": 17})
    service = S3SyncService(config)

    assert captured["service_name"] == "s3"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    boto_config = kwargs.get("config")
    assert getattr(boto_config, "max_pool_connections", None) == 17
    assert service._semaphore._value == 17  # noqa: SLF001


@pytest.mark.asyncio
async def test_download_changes_isolates_per_file_failures(
    monkeypatch: pytest.MonkeyPatch,
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """One failed download should not abort the entire batch."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    success_hash = compute_hash("ok")
    fail_hash = compute_hash("fail")
    delta = SyncDelta(
        from_version="0.0.0",
        to_version="4.0.5",
        module=SpringModule.BOOT,
        changes=[
            FileChange(
                path="ok/document.md",
                change_type=ChangeType.ADDED,
                new_hash=success_hash,
                size_bytes=10,
            ),
            FileChange(
                path="fail/document.md",
                change_type=ChangeType.ADDED,
                new_hash=fail_hash,
                size_bytes=5,
            ),
        ],
    )
    fail_key = sync_config.get_s3_key("spring-boot", "4.0.5", "fail/document.md")

    async def fake_download_file(s3_key: str, expected_hash: str | None = None) -> DownloadResult:
        del expected_hash
        if s3_key == fail_key:
            raise RuntimeError("simulated download failure")
        return DownloadResult(
            s3_key=s3_key,
            local_path=sync_config.get_local_path(s3_key),
            size_bytes=10,
            content_hash=success_hash,
            success=True,
        )

    monkeypatch.setattr(service, "download_file", fake_download_file)

    results = await service.download_changes(
        delta=delta,
        module="spring-boot",
        version="4.0.5",
    )

    assert len(results) == 2
    assert sum(1 for r in results if r.success) == 1
    assert sum(1 for r in results if not r.success) == 1
    failed = next(r for r in results if not r.success)
    assert failed.s3_key == fail_key
    assert failed.error is not None


@pytest.mark.asyncio
async def test_find_latest_snapshot_pair_uses_latest_common_date(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Snapshot discovery should select latest token shared by both archive families."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_db_snapshots_prefix()
    chroma_shared_key = f"{prefix}/chroma_db_2026_04_06.zip"
    sqlite_shared_key = f"{prefix}/sqlite_metadata_2026_04_06.zip"
    chroma_newer_only_key = f"{prefix}/chroma_db_2026_04_07.zip"

    for key, payload in (
        (chroma_shared_key, _zip_bytes({"spring_docs/index.bin": b"shared-chroma"})),
        (sqlite_shared_key, _zip_bytes({sync_config.db_filename: b"shared-sqlite"})),
        (chroma_newer_only_key, _zip_bytes({"spring_docs/index.bin": b"newer-chroma"})),
    ):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    selection = await service.find_latest_snapshot_pair()

    assert selection.snapshot_token == "2026_04_06"
    assert selection.chroma_key == chroma_shared_key
    assert selection.sqlite_key == sqlite_shared_key


@pytest.mark.asyncio
async def test_find_latest_snapshot_pair_falls_back_to_spring_docs_prefix(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Snapshot discovery should work when env prefix differs from canonical spring-docs."""
    mismatch_config = sync_config.model_copy(update={"s3_prefix": "docs"})
    service = S3SyncService(mismatch_config, s3_client=mock_s3)
    canonical_prefix = "spring-docs/db-snapshots"
    chroma_key = f"{canonical_prefix}/chroma_db_2026_04_07.zip"
    sqlite_key = f"{canonical_prefix}/sqlite_metadata_2026_04_07.zip"

    for key, payload in (
        (chroma_key, _zip_bytes({"spring_docs/index.bin": b"canonical-chroma"})),
        (sqlite_key, _zip_bytes({sync_config.db_filename: b"canonical-sqlite"})),
    ):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    selection = await service.find_latest_snapshot_pair()

    assert selection.snapshot_token == "2026_04_07"
    assert selection.chroma_key == chroma_key
    assert selection.sqlite_key == sqlite_key


@pytest.mark.asyncio
async def test_find_latest_snapshot_pair_prefers_namespaced_scope_over_root_fallback(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Namespaced snapshot keys must win over newer root-level fallback keys."""
    mismatch_config = sync_config.model_copy(update={"s3_prefix": "docs"})
    service = S3SyncService(mismatch_config, s3_client=mock_s3)

    namespaced_prefix = "spring-docs/db-snapshots/bge-m3-main"
    root_prefix = "spring-docs/db-snapshots"

    # Correct namespace snapshot pair (older token).
    namespaced_chroma = f"{namespaced_prefix}/chroma_db_2026_04_07.zip"
    namespaced_sqlite = f"{namespaced_prefix}/sqlite_metadata_2026_04_07.zip"
    # Root fallback pair with newer token should be ignored when namespace exists.
    root_chroma = f"{root_prefix}/chroma_db_2026_04_09.zip"
    root_sqlite = f"{root_prefix}/sqlite_metadata_2026_04_09.zip"

    for key, payload in (
        (namespaced_chroma, _zip_bytes({"spring_docs/index.bin": b"namespaced-chroma"})),
        (namespaced_sqlite, _zip_bytes({sync_config.db_filename: b"namespaced-sqlite"})),
        (root_chroma, _zip_bytes({"spring_docs/index.bin": b"root-chroma"})),
        (root_sqlite, _zip_bytes({sync_config.db_filename: b"root-sqlite"})),
    ):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    selection = await service.find_latest_snapshot_pair()

    assert selection.snapshot_token == "2026_04_07"
    assert selection.chroma_key == namespaced_chroma
    assert selection.sqlite_key == namespaced_sqlite


@pytest.mark.asyncio
async def test_download_latest_db_snapshots_applies_and_cleans_archives(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """snapshot-download should atomically activate latest shared snapshots."""
    _seed_local_datastores(sync_config)
    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_db_snapshots_prefix()

    sqlite_bytes = b"sqlite-restored-content"
    chroma_bytes = b"vector-restored-content"
    sqlite_payload = _zip_bytes({sync_config.db_filename: sqlite_bytes})
    chroma_payload = _zip_bytes({"spring_docs/index.bin": chroma_bytes})

    sqlite_key = f"{prefix}/sqlite_metadata_2026_04_08.zip"
    chroma_key = f"{prefix}/chroma_db_2026_04_08.zip"

    for key, payload in ((sqlite_key, sqlite_payload), (chroma_key, chroma_payload)):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    result = await service.download_latest_db_snapshots(cleanup_local_archives=True)

    assert result.success
    assert result.snapshot_token == "2026_04_08"
    assert result.sqlite_snapshot == "sqlite_metadata_2026_04_08.zip"
    assert result.chroma_snapshot == "chroma_db_2026_04_08.zip"
    assert sync_config.db_path.read_bytes() == sqlite_bytes
    assert (sync_config.chroma_dir / "spring_docs" / "index.bin").read_bytes() == chroma_bytes
    assert not (sync_config.packs_dir / "sqlite_metadata_2026_04_08.zip").exists()
    assert not (sync_config.packs_dir / "chroma_db_2026_04_08.zip").exists()


@pytest.mark.asyncio
async def test_download_latest_db_snapshots_restores_bm25_when_present(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """snapshot-download should restore bm25_index.pkl when bundled in sqlite snapshot."""
    _seed_local_datastores(sync_config)
    old_bm25 = sync_config.local_data_dir / "bm25_index.pkl"
    old_bm25.write_bytes(b"bm25-old-content")

    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_db_snapshots_prefix()

    sqlite_payload = _zip_bytes(
        {
            sync_config.db_filename: b"sqlite-restored-content",
            "bm25_index.pkl": b"bm25-restored-content",
        }
    )
    chroma_payload = _zip_bytes({"spring_docs/index.bin": b"vector-restored-content"})
    sqlite_key = f"{prefix}/sqlite_metadata_2026_04_11.zip"
    chroma_key = f"{prefix}/chroma_db_2026_04_11.zip"

    for key, payload in ((sqlite_key, sqlite_payload), (chroma_key, chroma_payload)):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    result = await service.download_latest_db_snapshots(cleanup_local_archives=True)

    assert result.success
    assert (sync_config.local_data_dir / "bm25_index.pkl").read_bytes() == b"bm25-restored-content"


@pytest.mark.asyncio
async def test_download_latest_db_snapshots_keeps_existing_bm25_if_missing_in_snapshot(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Older sqlite snapshots without bm25 file should keep existing local BM25 file."""
    _seed_local_datastores(sync_config)
    bm25_path = sync_config.local_data_dir / "bm25_index.pkl"
    bm25_path.write_bytes(b"bm25-existing-content")

    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_db_snapshots_prefix()

    sqlite_payload = _zip_bytes({sync_config.db_filename: b"sqlite-restored-content"})
    chroma_payload = _zip_bytes({"spring_docs/index.bin": b"vector-restored-content"})
    sqlite_key = f"{prefix}/sqlite_metadata_2026_04_12.zip"
    chroma_key = f"{prefix}/chroma_db_2026_04_12.zip"

    for key, payload in ((sqlite_key, sqlite_payload), (chroma_key, chroma_payload)):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    result = await service.download_latest_db_snapshots(cleanup_local_archives=True)

    assert result.success
    assert bm25_path.read_bytes() == b"bm25-existing-content"


@pytest.mark.asyncio
async def test_download_latest_db_snapshots_rolls_back_on_invalid_archive(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Activation failure should keep currently-live sqlite/chroma content untouched."""
    original_db, original_chroma = _seed_local_datastores(sync_config)
    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_db_snapshots_prefix()

    sqlite_key = f"{prefix}/sqlite_metadata_2026_04_09.zip"
    chroma_key = f"{prefix}/chroma_db_2026_04_09.zip"
    sqlite_payload = _zip_bytes({sync_config.db_filename: b"sqlite-new-content"})
    invalid_chroma_payload = b"not-a-valid-zip"

    for key, payload in ((sqlite_key, sqlite_payload), (chroma_key, invalid_chroma_payload)):
        mock_s3.put_object(
            Bucket=sync_config.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="application/zip",
            Metadata={"content-hash": compute_hash(payload)},
        )

    result = await service.download_latest_db_snapshots(cleanup_local_archives=True)

    assert not result.success
    assert result.error is not None
    assert sync_config.db_path.read_bytes() == original_db
    assert (
        sync_config.chroma_dir / "spring_docs" / "index.bin"
    ).read_bytes() == original_chroma


@pytest.mark.asyncio
async def test_find_latest_snapshot_pair_fails_without_common_date(
    mock_s3: Any,
    sync_config: SyncConfig,
) -> None:
    """Snapshot discovery should fail clearly when no shared token exists."""
    service = S3SyncService(sync_config, s3_client=mock_s3)
    prefix = sync_config.get_db_snapshots_prefix()

    mock_s3.put_object(
        Bucket=sync_config.s3_bucket,
        Key=f"{prefix}/chroma_db_2026_04_10.zip",
        Body=_zip_bytes({"spring_docs/index.bin": b"only-chroma"}),
        ContentType="application/zip",
    )
    mock_s3.put_object(
        Bucket=sync_config.s3_bucket,
        Key=f"{prefix}/sqlite_metadata_2026_04_09.zip",
        Body=_zip_bytes({sync_config.db_filename: b"only-sqlite"}),
        ContentType="application/zip",
    )

    with pytest.raises(ValueError, match="No common snapshot date found"):
        await service.find_latest_snapshot_pair()
