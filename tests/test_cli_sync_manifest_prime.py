"""Tests for manifest-prime CLI behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest

from everspring_mcp import main as cli_main
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

    monkeypatch.setattr(cli_main, "S3SyncService", FakeService)
    monkeypatch.setattr(cli_main, "_load_sync_targets_from_matrix", lambda _path: targets)

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

    exit_code = await cli_main._run_manifest_prime(args, config)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert counters["max_active"] == 5
    assert payload["parallel_jobs"] == 5
    assert payload["targets"] == len(targets)
    assert payload["uploaded"] == len(targets)


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

    monkeypatch.setattr(cli_main, "_run_manifest_sync", fake_run_manifest_sync)
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

    exit_code = await cli_main._run_sync(args)

    assert exit_code == 0
    assert captured["parallel_jobs"] == 9
    assert captured["download_concurrency"] == 9


def test_sync_parser_accepts_snapshot_download_mode() -> None:
    """sync parser should accept snapshot-download mode."""
    parser = cli_main._build_parser()
    args = parser.parse_args(["sync", "--mode", "snapshot-download"])
    assert args.mode == "snapshot-download"


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

        async def download_latest_snapshots(self) -> SnapshotDownloadResult:
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

    monkeypatch.setattr(cli_main, "SyncOrchestrator", FakeOrchestrator)
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
        data_dir=None,
        json=False,
    )

    exit_code = await cli_main._run_sync(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Sync complete." in captured.out
    assert "Vector DB is now live and ready for search." in captured.out
