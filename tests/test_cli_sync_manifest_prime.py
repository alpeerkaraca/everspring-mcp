"""Tests for manifest-prime CLI behavior."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from everspring_mcp import main as cli_main
from everspring_mcp.cli import index as index_cli
from everspring_mcp.cli import sync as sync_cli
from everspring_mcp.cli import mcp as mcp_cli
from everspring_mcp.cli import model_cache as cache_cli
from everspring_mcp.cli import utils as cli_utils
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.s3_sync import ManifestBuildResult, SnapshotDownloadResult


def test_sync_parser_manifest_prime_parallel_jobs_default() -> None:
    """sync parser should expose --parallel-jobs with default 5."""
    parser = cli_main._build_parser()
    args = parser.parse_args(
        [
            "sync",
            "--mode",
            "manifest-prime",
            "--module",
            "spring-boot",
            "--version",
            "4.0.5",
        ]
    )
    assert args.parallel_jobs == 5


@pytest.mark.asyncio
async def test_run_manifest_prime_honors_parallel_jobs_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """manifest-prime should process targets concurrently up to --parallel-jobs."""
    counters: dict[str, int] = {"active": 0, "max_active": 0}

    class FakeService:
        def __init__(self, _config: SyncConfig) -> None:
            pass

        async def discover_all_targets(self) -> list[tuple[str, str, str | None]]:
            return targets

        async def ensure_manifest(
            self,
            module: str,
            version: str,
            submodule: str | None = None,
            force: bool = False,
        ) -> ManifestBuildResult:
            del force
            counters["active"] += 1
            counters["max_active"] = max(counters["max_active"], counters["active"])
            await asyncio.sleep(0.02)
            counters["active"] -= 1
            return ManifestBuildResult(
                module=module,
                version=version,
                submodule=submodule,
                manifest_key=f"test-docs/raw-data/{module}/{version}/manifest.json",
                status="uploaded",
                file_count=2,
                total_size_bytes=42,
            )

    targets = [
        ("spring-boot", "4.0.5", None),
        ("spring-framework", "7.0.6", None),
        ("spring-security", "7.0.4", None),
        ("spring-ai", "1.1.4", None),
        ("spring-cloud", "5.0.1", "netflix"),
        ("spring-cloud", "5.0.1", "gateway"),
        ("spring-data", "5.0.4", "jpa"),
    ]

    monkeypatch.setattr(sync_cli, "S3SyncService", FakeService)

    args = argparse.Namespace(
        all=True,
        module=None,
        version=None,
        submodule=None,
        force=False,
        json=True,
        parallel_jobs=5,
    )
    config = SyncConfig(
        s3_bucket="test-bucket",
        s3_region="us-east-1",
        s3_prefix="test-docs",
        local_data_dir=tmp_path,
    )

    exit_code = await sync_cli._run_manifest_prime(args, config)
    capsys.readouterr()

    assert exit_code == 0
    assert counters["max_active"] == 5


@pytest.mark.asyncio
async def test_run_sync_manifest_passes_parallel_jobs_to_sync_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """manifest mode should propagate --parallel-jobs to sync config and manifest runner."""
    captured: dict[str, int] = {}

    async def fake_run_manifest_sync(_args: argparse.Namespace, config: SyncConfig) -> int:
        captured["parallel_jobs"] = config.parallel_jobs
        captured["download_concurrency"] = config.download_concurrency
        return 0

    def fake_from_env(cls: type[SyncConfig]) -> SyncConfig:
        return cls(
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            s3_prefix="test-docs",
            local_data_dir=tmp_path,
        )

    monkeypatch.setattr(sync_cli, "_run_manifest_sync", fake_run_manifest_sync)
    monkeypatch.setattr(
        SyncConfig,
        "from_env",
        classmethod(fake_from_env),
    )

    args = argparse.Namespace(
        mode="manifest",
        force=False,
        all=False,
        module="spring-boot",
        version="4.0.5",
        submodule=None,
        parallel_jobs=9,
        s3_bucket=None,
        s3_region=None,
        s3_prefix=None,
        data_dir=None,
        json=True,
    )

    exit_code = await sync_cli._run_sync(args)

    assert exit_code == 0
    assert captured["parallel_jobs"] == 9
    assert captured["download_concurrency"] == 9


def test_sync_parser_accepts_snapshot_download_mode() -> None:
    """sync parser should accept snapshot-download mode."""
    parser = cli_main._build_parser()
    args = parser.parse_args(["sync", "--mode", "snapshot-download"])
    assert args.mode == "snapshot-download"


def test_sync_parser_accepts_snapshot_namespace_flags() -> None:
    parser = cli_main._build_parser()
    args = parser.parse_args(
        ["sync", "--mode", "snapshot-download", "--snapshot-model", "BAAI/bge-m3", "--snapshot-tier", "main"]
    )
    assert args.snapshot_model == "BAAI/bge-m3"
    assert args.snapshot_tier == "main"


@pytest.mark.asyncio
async def test_run_sync_snapshot_download_uses_orchestrator_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """snapshot-download mode should print ready summary and return success code."""

    class FakeOrchestrator:
        def __init__(self, config: SyncConfig) -> None:
            self.config = config

        async def __aenter__(self) -> FakeOrchestrator:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def download_latest_snapshots(
            self,
            model_name: str | None = None,
            tier: str | None = None,
        ) -> SnapshotDownloadResult:
            del model_name, tier
            return SnapshotDownloadResult(
                snapshot_token="2026_04_08",
                chroma_snapshot="chroma_db_2026_04_08.zip",
                sqlite_snapshot="sqlite_metadata_2026_04_08.zip",
                bytes_downloaded=740 * 1024 * 1024,
                success=True,
            )

    def fake_from_env(cls: type[SyncConfig]) -> SyncConfig:
        return cls(
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            s3_prefix="test-docs",
            local_data_dir=tmp_path,
        )

    monkeypatch.setattr(sync_cli, "SyncOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        SyncConfig,
        "from_env",
        classmethod(fake_from_env),
    )

    args = argparse.Namespace(
        mode="snapshot-download",
        force=False,
        all=False,
        module=None,
        version=None,
        submodule=None,
        parallel_jobs=5,
        s3_bucket=None,
        s3_region=None,
        s3_prefix=None,
        snapshot_model=None,
        snapshot_tier=None,
        data_dir=None,
        json=False,
    )

    exit_code = await sync_cli._run_sync(args)
    capsys.readouterr()

    assert exit_code == 0


@pytest.mark.asyncio
async def test_run_sync_snapshot_upload_defaults_chroma_dir_to_tier_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    class FakeService:
        def __init__(self, config: SyncConfig) -> None:
            captured["chroma_dir"] = str(config.chroma_dir)
            captured["model_name"] = config.model_name
            captured["model_tier"] = config.model_tier

        async def upload_db_snapshots(
            self,
            snapshot_date=None,
            cleanup_local_archives: bool = False,
            model_name: str | None = None,
            tier: str | None = None,
        ) -> list[object]:
            del snapshot_date, cleanup_local_archives, model_name, tier
            return []

    def fake_from_env(cls: type[SyncConfig]) -> SyncConfig:
        return cls(
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            s3_prefix="test-docs",
            local_data_dir=tmp_path,
        )

    monkeypatch.delenv("EVERSPRING_CHROMA_DIR", raising=False)
    monkeypatch.setattr(sync_cli, "S3SyncService", FakeService)
    monkeypatch.setattr(
        SyncConfig,
        "from_env",
        classmethod(fake_from_env),
    )

    args = argparse.Namespace(
        mode="snapshot-upload",
        force=False,
        all=False,
        module=None,
        version=None,
        submodule=None,
        parallel_jobs=5,
        s3_bucket=None,
        s3_region=None,
        s3_prefix=None,
        snapshot_model="BAAI/bge-small-en-v1.5",
        snapshot_tier="xslim",
        data_dir=None,
        json=True,
    )

    exit_code = await sync_cli._run_sync(args)

    assert exit_code == 0
    assert captured["model_name"] == "BAAI/bge-small-en-v1.5"
    assert captured["model_tier"] == "xslim"
    assert captured["chroma_dir"] == str(Path.home() / ".everspring" / "chroma-xslim-bge-small-en-v1-5")


@pytest.mark.asyncio
async def test_run_sync_snapshot_upload_all_uploads_all_default_tiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uploads: list[tuple[str, str, str]] = []

    class FakeService:
        def __init__(self, config: SyncConfig) -> None:
            self.config = config

        async def upload_db_snapshots(
            self,
            snapshot_date=None,
            cleanup_local_archives: bool = False,
            model_name: str | None = None,
            tier: str | None = None,
        ) -> list[object]:
            del snapshot_date, cleanup_local_archives
            assert model_name is not None
            assert tier is not None
            uploads.append((tier, model_name, str(self.config.chroma_dir)))
            return []

    def fake_from_env(cls: type[SyncConfig]) -> SyncConfig:
        return cls(
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            s3_prefix="test-docs",
            local_data_dir=tmp_path,
        )

    monkeypatch.setattr(sync_cli, "S3SyncService", FakeService)
    monkeypatch.setattr(
        SyncConfig,
        "from_env",
        classmethod(fake_from_env),
    )

    args = argparse.Namespace(
        mode="snapshot-upload",
        force=False,
        all=True,
        module=None,
        version=None,
        submodule=None,
        parallel_jobs=5,
        s3_bucket=None,
        s3_region=None,
        s3_prefix=None,
        snapshot_model=None,
        snapshot_tier=None,
        data_dir=None,
        json=True,
    )

    exit_code = await sync_cli._run_sync(args)

    assert exit_code == 0
    assert uploads == [
        ("main", "BAAI/bge-m3", str(Path.home() / ".everspring" / "chroma-main-bge-m3")),
        (
            "slim",
            "BAAI/bge-base-en-v1.5",
            str(Path.home() / ".everspring" / "chroma-slim-bge-base-en-v1-5"),
        ),
        (
            "xslim",
            "BAAI/bge-small-en-v1.5",
            str(Path.home() / ".everspring" / "chroma-xslim-bge-small-en-v1-5"),
        ),
    ]


@pytest.mark.asyncio
async def test_run_sync_snapshot_upload_all_rejects_snapshot_namespace_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_from_env(cls: type[SyncConfig]) -> SyncConfig:
        return cls(
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            s3_prefix="test-docs",
            local_data_dir=tmp_path,
        )

    monkeypatch.setattr(
        SyncConfig,
        "from_env",
        classmethod(fake_from_env),
    )

    args = argparse.Namespace(
        mode="snapshot-upload",
        force=False,
        all=True,
        module=None,
        version=None,
        submodule=None,
        parallel_jobs=5,
        s3_bucket=None,
        s3_region=None,
        s3_prefix=None,
        snapshot_model="BAAI/bge-m3",
        snapshot_tier=None,
        data_dir=None,
        json=True,
    )

    with pytest.raises(SystemExit, match="--all cannot be combined"):
        await sync_cli._run_sync(args)
