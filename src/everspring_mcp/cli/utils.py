from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from everspring_mcp.models.content import ContentType
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.sync.orchestrator import SyncOrchestrator
from everspring_mcp.sync.s3_sync import S3SyncService
from everspring_mcp.vector.config import VectorConfig

logger = logging.getLogger("everspring_mcp")
console = Console(stderr=True)

def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def _render_kv_panel(
    title: str, values: dict[str, Any], *, border_style: str = "cyan"
) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for key, value in values.items():
        grid.add_row(key.replace("_", " ").title(), _format_cell(value))
    console.print(Panel(grid, title=title, border_style=border_style))

def _render_error_table(errors: list[str]) -> None:
    if not errors:
        return
    table = Table(title="Errors", header_style="bold red")
    table.add_column("#", style="bold")
    table.add_column("Message")
    for idx, error in enumerate(errors, 1):
        table.add_row(str(idx), error)
    console.print(table)

def _render_status_rows(title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print(
            Panel("No status data available.", title=title, border_style="yellow")
        )
        return
    preferred_columns = [
        "module",
        "submodule",
        "version",
        "status",
        "files_added",
        "files_modified",
        "files_removed",
        "bytes_downloaded",
    ]
    all_columns = {key for row in rows for key in row}
    columns = [col for col in preferred_columns if col in all_columns]
    if not columns:
        columns = sorted(all_columns)
    table = Table(title=title, header_style="bold blue")
    for column in columns:
        table.add_column(column.replace("_", " ").title())
    for row in rows:
        table.add_row(*[_format_cell(row.get(column)) for column in columns])
    console.print(table)

def _render_search_results(results: list[Any]) -> None:
    if not results:
        console.print(Panel("No results found.", title="Search", border_style="yellow"))
        return

    summary = Table(title="Spring Docs Search Results", header_style="bold magenta")
    summary.add_column("#", style="bold")
    summary.add_column("Title")
    summary.add_column("Module")
    summary.add_column("Version")
    summary.add_column("Score", justify="right")
    summary.add_column("URL", overflow="fold")
    for idx, result in enumerate(results, 1):
        summary.add_row(
            str(idx),
            result.title,
            result.module,
            f"{result.version_major}.{result.version_minor}",
            f"{result.score:.4f}",
            result.url,
        )
    console.print(summary)

    for idx, result in enumerate(results, 1):
        rank_info = (
            f"Dense rank: #{result.dense_rank if result.dense_rank is not None else '-'} | "
            f"Sparse rank: #{result.sparse_rank if result.sparse_rank is not None else '-'}"
        )
        console.print(
            Panel(
                f"{rank_info}\n\n{result.content}",
                title=f"[{idx}] {result.title}",
                subtitle=result.url,
                border_style="cyan",
            )
        )

def _parse_module(value: str) -> SpringModule:
    normalized = value.strip().lower()
    for module in SpringModule:
        if module.value == normalized:
            return module
    raise argparse.ArgumentTypeError(f"Invalid module: {value}")

def _parse_content_type(value: str) -> ContentType:
    normalized = value.strip().lower()
    for content_type in ContentType:
        if content_type.value == normalized:
            return content_type
    raise argparse.ArgumentTypeError(f"Invalid content type: {value}")

def _parse_version(value: str, module: SpringModule) -> SpringVersion:
    try:
        return SpringVersion.parse(module, value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected integer value, got: {value}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a value greater than 0, got: {value}"
        )
    return parsed

def _model_slug(model_name: str) -> str:
    model_tail = model_name.strip().split("/")[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", model_tail.lower()).strip("-")
    return slug or "model"

def _tier_chroma_dir(tier: str, model_name: str) -> Path:
    return Path.home() / ".everspring" / f"chroma-{tier}-{_model_slug(model_name)}"

def _snapshot_state_path(data_dir: Path) -> Path:
    return data_dir / ".snapshot_state.json"

def _load_snapshot_state(data_dir: Path) -> dict[str, str]:
    state_path = _snapshot_state_path(data_dir)
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    state: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            state[key] = value
    return state

def _save_snapshot_state(data_dir: Path, state: dict[str, str]) -> None:
    state_path = _snapshot_state_path(data_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _restart_current_process() -> None:
    env = os.environ.copy()
    env["EVERSPRING_AUTO_SNAPSHOT_RESTARTED"] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "everspring_mcp.main", *sys.argv[1:]],
        env,
    )

async def _auto_refresh_runtime_snapshots(
    *,
    model_name: str,
    tier: str,
    data_dir: Path,
) -> None:
    if os.environ.get("EVERSPRING_AUTO_SNAPSHOT_RESTARTED") == "1":
        return

    chroma_override = os.environ.get(VectorConfig.ENV_CHROMA_DIR)
    sync_updates: dict[str, Any] = {
        "local_data_dir": data_dir,
        "model_name": model_name,
        "model_tier": tier,
        "chroma_subdir": chroma_override or str(_tier_chroma_dir(tier, model_name)),
    }
    sync_config = SyncConfig.from_env().model_copy(update=sync_updates)
    sync_config.ensure_directories()
    service = S3SyncService(sync_config)
    try:
        selection = await service.find_latest_snapshot_pair(
            model_name=model_name,
            tier=tier,
        )
    except ValueError:
        return

    namespace = sync_config.get_snapshot_namespace(model_name=model_name, tier=tier)
    selected_state = (
        f"{selection.snapshot_token}|{selection.chroma_key}|{selection.sqlite_key}"
    )
    state = _load_snapshot_state(data_dir)
    if state.get(namespace) == selected_state:
        return

    async with SyncOrchestrator(sync_config, s3_service=service) as orchestrator:
        download_result = await orchestrator.download_latest_snapshots(
            model_name=model_name,
            tier=tier,
        )
    if not download_result.success or not download_result.snapshot_token:
        logger.warning(
            "Auto snapshot refresh failed for %s: %s",
            namespace,
            download_result.error or "unknown error",
        )
        return

    state[namespace] = selected_state
    _save_snapshot_state(data_dir, state)
    logger.info(
        "Applied newer snapshot %s for %s; restarting process",
        download_result.snapshot_token,
        namespace,
    )
    _restart_current_process()

async def resolve_vector_config(args: argparse.Namespace) -> VectorConfig:
    from everspring_mcp.vector.embeddings import default_model_for_tier

    config = VectorConfig.from_env()
    updates: dict[str, Any] = {}

    if getattr(args, "data_dir", None):
        updates["data_dir"] = Path(args.data_dir)
    if getattr(args, "db_filename", None):
        updates["db_filename"] = args.db_filename
    if getattr(args, "docs_subdir", None):
        updates["docs_subdir"] = args.docs_subdir
    if getattr(args, "chroma_dir", None):
        updates["chroma_dir"] = Path(args.chroma_dir)
    if getattr(args, "collection", None):
        updates["collection_name"] = args.collection

    selected_tier = getattr(args, "tier", "main")
    updates["embedding_tier"] = selected_tier
    resolved_model = getattr(args, "embed_model", None) or default_model_for_tier(selected_tier)
    updates["embedding_model"] = resolved_model

    if not updates.get("chroma_dir") and not os.environ.get(VectorConfig.ENV_CHROMA_DIR):
        updates["chroma_dir"] = _tier_chroma_dir(selected_tier, resolved_model)

    if updates:
        config = config.model_copy(update=updates)

    await _auto_refresh_runtime_snapshots(
        model_name=config.embedding_model,
        tier=config.embedding_tier,
        data_dir=config.data_dir,
    )
    return config
