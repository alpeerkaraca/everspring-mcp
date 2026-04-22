from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.orchestrator import SyncOrchestrator
from everspring_mcp.sync.s3_sync import S3SyncService
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import default_model_for_tier
from everspring_mcp.cli.utils import (
    _tier_chroma_dir,
    _render_kv_panel,
    _render_error_table,
    _render_status_rows,
)

logger = logging.getLogger("everspring_mcp")

def _load_sync_targets_from_matrix(
    matrix_path: Path,
) -> list[tuple[str, str, str | None]]:
    if not matrix_path.exists():
        raise SystemExit(f"CSV not found: {matrix_path}")

    import csv

    targets: list[tuple[str, str, str | None]] = []
    seen: set[tuple[str, str, str | None]] = set()
    with matrix_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            module = (row.get("module") or "").strip()
            version = (row.get("version") or "").strip()
            submodule = (row.get("submodule") or "").strip() or None
            if not module or not version:
                continue
            key = (module, version, submodule)
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
    return targets

async def _run_manifest_sync(args: argparse.Namespace, config: SyncConfig) -> int:
    async with SyncOrchestrator(config) as orchestrator:
        if args.all:
            targets = _load_sync_targets_from_matrix(
                Path("config") / "module_submodule_urls.csv"
            )
            results = []
            for module, version, submodule in targets:
                res = await orchestrator.sync_module(
                    module=module,
                    version=version,
                    submodule=submodule,
                    force=args.force,
                )
                results.append(res)

            summary = {
                "mode": "manifest",
                "targets": len(results),
                "completed": sum(1 for r in results if r.status.value == "completed"),
                "failed": sum(1 for r in results if r.status.value == "failed"),
                "files_added": sum(r.files_added for r in results),
                "files_modified": sum(r.files_modified for r in results),
                "files_removed": sum(r.files_removed for r in results),
                "bytes_downloaded": sum(r.bytes_downloaded for r in results),
                "errors": [
                    f"{r.module}:{r.version} -> {err}"
                    for r in results
                    for err in r.errors
                ],
            }
            if args.json:
                logger.info(json.dumps(summary, indent=2))
            else:
                _render_kv_panel(
                    "Sync Summary (manifest)",
                    {
                        "targets": summary["targets"],
                        "completed": summary["completed"],
                        "failed": summary["failed"],
                        "files_added": summary["files_added"],
                        "files_modified": summary["files_modified"],
                        "files_removed": summary["files_removed"],
                        "bytes_downloaded": summary["bytes_downloaded"],
                    },
                )
                _render_error_table(summary["errors"])
            return 0 if summary["failed"] == 0 else 1

        result = await orchestrator.sync_module(
            module=args.module,
            version=args.version,
            submodule=args.submodule,
            force=args.force,
        )

    if args.json:
        logger.info(result.model_dump_json(indent=2))
    else:
        _render_kv_panel(
            "Sync Status (manifest)",
            {
                "status": result.status.value,
                "module": result.module,
                "version": result.version,
                "submodule": result.submodule or "",
                "files_added": result.files_added,
                "files_modified": result.files_modified,
                "files_removed": result.files_removed,
                "bytes_downloaded": result.bytes_downloaded,
            },
        )
        _render_error_table(result.errors)
    return 0 if result.status.value == "completed" else 1

async def _run_manifest_prime(args: argparse.Namespace, config: SyncConfig) -> int:
    s3_service = S3SyncService(config)
    parallel_jobs = config.parallel_jobs

    if args.all:
        targets = _load_sync_targets_from_matrix(
            Path("config") / "module_submodule_urls.csv"
        )
    else:
        targets = [(args.module, args.version, args.submodule)]

    semaphore = asyncio.Semaphore(parallel_jobs)

    async def _prime_target(module: str, version: str, submodule: str | None):
        async with semaphore:
            return await s3_service.ensure_manifest(
                module=module,
                version=version,
                submodule=submodule,
                force=args.force,
            )

    tasks = [
        _prime_target(module, version, submodule)
        for module, version, submodule in targets
    ]
    results = await asyncio.gather(*tasks)

    summary = {
        "mode": "manifest-prime",
        "parallel_jobs": parallel_jobs,
        "targets": len(results),
        "uploaded": sum(1 for r in results if r.status == "uploaded"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "files_covered": sum(r.file_count for r in results if r.status != "failed"),
        "total_size_bytes": sum(
            r.total_size_bytes for r in results if r.status != "failed"
        ),
        "results": [r.model_dump() for r in results],
        "errors": [
            f"{r.module}:{r.version}{'/' + r.submodule if r.submodule else ''} -> {r.error}"
            for r in results
            if r.status == "failed" and r.error
        ],
    }

    if args.json:
        logger.info(json.dumps(summary, indent=2, default=str))
    else:
        _render_kv_panel(
            "Manifest Prime Summary",
            {
                "targets": summary["targets"],
                "uploaded": summary["uploaded"],
                "skipped": summary["skipped"],
                "failed": summary["failed"],
                "files_covered": summary["files_covered"],
                "total_size_bytes": summary["total_size_bytes"],
                "parallel_jobs": summary["parallel_jobs"],
            },
        )
        _render_error_table(summary["errors"])
    return 0 if summary["failed"] == 0 else 1

async def _run_sync(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    if args.parallel_jobs < 1:
        raise SystemExit("--parallel-jobs must be >= 1")
    updates: dict[str, Any] = {}
    if args.s3_bucket:
        updates["s3_bucket"] = args.s3_bucket
    if args.s3_region:
        updates["s3_region"] = args.s3_region
    if args.s3_prefix:
        updates["s3_prefix"] = args.s3_prefix
    if args.data_dir:
        updates["local_data_dir"] = Path(args.data_dir)
    snapshot_tier = getattr(args, "snapshot_tier", None)
    snapshot_model = getattr(args, "snapshot_model", None)
    if snapshot_tier:
        updates["model_tier"] = snapshot_tier
    if snapshot_model:
        updates["model_name"] = snapshot_model
    if args.mode in {"manifest", "manifest-prime"}:
        updates["parallel_jobs"] = args.parallel_jobs
        updates["download_concurrency"] = args.parallel_jobs
    if args.mode in {"snapshot-upload", "snapshot-download"} and not (
        args.mode == "snapshot-upload" and args.all
    ):
        effective_snapshot_tier = updates.get("model_tier", config.model_tier)
        effective_snapshot_model = updates.get("model_name", config.model_name)
        snapshot_chroma_override = os.environ.get(VectorConfig.ENV_CHROMA_DIR)
        updates["chroma_subdir"] = snapshot_chroma_override or str(
            _tier_chroma_dir(effective_snapshot_tier, effective_snapshot_model)
        )
    if updates:
        config = config.model_copy(update=updates)

    if args.mode in {"snapshot-upload", "snapshot-download"}:
        if args.force:
            raise SystemExit(
                "--force is only supported with --mode manifest or --mode manifest-prime"
            )
        if any([args.module, args.version, args.submodule]):
            raise SystemExit(
                f"--mode {args.mode} does not accept --module/--version/--submodule"
            )
        if args.mode == "snapshot-download" and args.all:
            raise SystemExit("--all is only supported with --mode snapshot-upload")
        if (
            args.mode == "snapshot-upload"
            and args.all
            and any([snapshot_tier, snapshot_model])
        ):
            raise SystemExit(
                "--all cannot be combined with --snapshot-model/--snapshot-tier for --mode snapshot-upload"
            )

        if args.mode == "snapshot-upload":
            upload_configs: list[SyncConfig] = []
            if args.all:
                for tier_name in ("main", "slim", "xslim"):
                    tier_model = default_model_for_tier(tier_name)
                    upload_configs.append(
                        config.model_copy(
                            update={
                                "model_tier": tier_name,
                                "model_name": tier_model,
                                "chroma_subdir": str(
                                    _tier_chroma_dir(tier_name, tier_model)
                                ),
                            }
                        )
                    )
            else:
                upload_configs.append(config)

            snapshot_results = []
            for upload_config in upload_configs:
                s3_service = S3SyncService(upload_config)
                snapshot_results.extend(
                    await s3_service.upload_db_snapshots(
                        model_name=upload_config.model_name,
                        tier=upload_config.model_tier,
                    )
                )

            summary = {
                "mode": args.mode,
                "scope": "all-tiers" if args.all else "single-namespace",
                "snapshot_namespaces": sorted(
                    {r.snapshot_namespace for r in snapshot_results}
                ),
                "snapshots": len(snapshot_results),
                "completed": sum(1 for r in snapshot_results if r.success),
                "failed": sum(1 for r in snapshot_results if not r.success),
                "bytes_uploaded": sum(
                    r.size_bytes for r in snapshot_results if r.success
                ),
                "results": [
                    {
                        "snapshot_name": r.snapshot_name,
                        "s3_key": r.s3_key,
                        "size_bytes": r.size_bytes,
                        "content_hash": r.content_hash,
                        "snapshot_namespace": r.snapshot_namespace,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in snapshot_results
                ],
                "errors": [
                    f"{r.snapshot_name} -> {r.error}"
                    for r in snapshot_results
                    if not r.success and r.error
                ],
            }

            if args.json:
                logger.info(json.dumps(summary, indent=2))
            else:
                _render_kv_panel(
                    "Snapshot Upload Summary",
                    {
                        "mode": args.mode,
                        "scope": summary["scope"],
                        "snapshots": summary["snapshots"],
                        "completed": summary["completed"],
                        "failed": summary["failed"],
                        "bytes_uploaded": summary["bytes_uploaded"],
                    },
                )
                _render_error_table(summary["errors"])
            return 0 if summary["failed"] == 0 else 1

        if args.mode == "snapshot-download":
            async with SyncOrchestrator(config) as orchestrator:
                snapshot_result = await orchestrator.download_latest_snapshots(
                    model_name=config.model_name,
                    tier=config.model_tier,
                )

            summary = {
                "mode": args.mode,
                "snapshot_namespace": snapshot_result.snapshot_namespace,
                "snapshot_token": snapshot_result.snapshot_token,
                "chroma_snapshot": snapshot_result.chroma_snapshot,
                "sqlite_snapshot": snapshot_result.sqlite_snapshot,
                "bytes_downloaded": snapshot_result.bytes_downloaded,
                "success": snapshot_result.success,
                "error": snapshot_result.error,
            }

            if args.json:
                logger.info(json.dumps(summary, indent=2))
            else:
                if snapshot_result.success:
                    downloaded_mb = snapshot_result.bytes_downloaded / (1024 * 1024)
                    _render_kv_panel(
                        "Snapshot Download Complete",
                        {
                            "namespace": summary["snapshot_namespace"],
                            "snapshot_token": summary["snapshot_token"],
                            "chroma_snapshot": summary["chroma_snapshot"],
                            "sqlite_snapshot": summary["sqlite_snapshot"],
                            "downloaded_mb": round(downloaded_mb, 1),
                        },
                    )
                else:
                    _render_kv_panel(
                        "Snapshot Download Failed",
                        {
                            "mode": args.mode,
                            "error": snapshot_result.error or "unknown error",
                        },
                        border_style="red",
                    )
            return 0 if snapshot_result.success else 1

    if args.all and any([args.module, args.version, args.submodule]):
        raise SystemExit("--all cannot be combined with --module/--version/--submodule")
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module")
    if not args.all and (not args.module or not args.version):
        raise SystemExit("sync requires --module and --version, or use --all")

    if args.mode == "manifest-prime":
        return await _run_manifest_prime(args, config)

    return await _run_manifest_sync(args, config)

async def _run_status(args: argparse.Namespace) -> int:
    config = SyncConfig.from_env()
    if args.data_dir:
        config = config.model_copy(update={"local_data_dir": Path(args.data_dir)})
    if args.all and any([args.module, args.version, args.submodule]):
        raise SystemExit("--all cannot be combined with --module/--version/--submodule")
    if args.submodule and not args.module:
        raise SystemExit("--submodule requires --module")
    if not args.all and (not args.module or not args.version):
        raise SystemExit("status requires --module and --version, or use --all")

    async with SyncOrchestrator(config) as orchestrator:
        if args.all:
            targets = await orchestrator.list_manifest_targets()
            statuses = []
            for target in targets:
                module = target["module"]
                version = target["version"]
                submodule = target["submodule"]
                status = await orchestrator.get_sync_status(
                    module=module,  # type: ignore[arg-type]
                    version=version,  # type: ignore[arg-type]
                    submodule=submodule,  # type: ignore[arg-type]
                )
                statuses.append(status)
            if args.json:
                logger.info(json.dumps(statuses, indent=2))
            else:
                _render_status_rows("Sync Status (all targets)", statuses)
            return 0

        status = await orchestrator.get_sync_status(
            args.module,
            args.version,
            submodule=args.submodule,
        )

    if args.json:
        logger.info(json.dumps(status, indent=2))
    else:
        _render_status_rows("Sync Status", [status])
    return 0

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    sync = subparsers.add_parser(
        "sync",
        help="Sync docs and snapshots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sync.add_argument(
        "--module", required=False, help="Spring module (e.g., spring-boot)"
    )
    sync.add_argument("--version", required=False, help="Version string (e.g., 4.0.5)")
    sync.add_argument(
        "--submodule", default=None, help="Optional submodule key (e.g., redis)"
    )
    sync.add_argument(
        "--all",
        action="store_true",
        help=(
            "For manifest modes: sync all discovered module/submodule/version targets from "
            "config\\module_submodule_urls.csv. For snapshot-upload: upload snapshots for all "
            "default embedding tiers."
        ),
    )
    sync.add_argument(
        "--mode",
        choices=[
            "manifest",
            "manifest-prime",
            "snapshot-upload",
            "snapshot-download",
        ],
        default="manifest",
        help=(
            "Sync mode (manifest downloads raw docs; manifest-prime pre-generates/uploads "
            "manifest.json files; snapshot-upload backs up local SQLite+Chroma; "
            "snapshot-download restores latest matching SQLite+Chroma snapshots)"
        ),
    )
    sync.add_argument(
        "--force",
        action="store_true",
        help="Force operation in manifest/manifest-prime modes",
    )
    sync.add_argument(
        "--parallel-jobs",
        type=int,
        default=5,
        help="Parallel workers for manifest download/priming",
    )
    sync.add_argument("--s3-bucket", default=None, help="S3 bucket override")
    sync.add_argument("--s3-region", default=None, help="S3 region override")
    sync.add_argument("--s3-prefix", default=None, help="S3 key prefix override")
    sync.add_argument(
        "--snapshot-model",
        default=None,
        help="Snapshot namespace model name override for snapshot-upload/download",
    )
    sync.add_argument(
        "--snapshot-tier",
        choices=["main", "slim", "xslim"],
        default=None,
        help="Snapshot namespace tier override for snapshot-upload/download",
    )
    sync.add_argument("--data-dir", default=None, help="Local data directory override")
    sync.add_argument("--json", action="store_true", help="Output JSON summary")
    sync.set_defaults(func=_run_sync)

    status = subparsers.add_parser(
        "status",
        help="Show sync status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    status.add_argument(
        "--module", required=False, help="Spring module (e.g., spring-boot)"
    )
    status.add_argument(
        "--version", required=False, help="Version string (e.g., 4.0.5)"
    )
    status.add_argument(
        "--submodule", default=None, help="Optional submodule key (e.g., redis)"
    )
    status.add_argument(
        "--all",
        action="store_true",
        help="Show status for all module/submodule/version entries in local manifest cache",
    )
    status.add_argument(
        "--data-dir", default=None, help="Local data directory override"
    )
    status.add_argument("--json", action="store_true", help="Output JSON summary")
    status.set_defaults(func=_run_status)
