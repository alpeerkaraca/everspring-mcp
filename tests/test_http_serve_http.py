"""Tests for HTTP transport runtime and Granian launch helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from anyio import ClosedResourceError
from starlette.testclient import TestClient

from everspring_mcp.http import serve_http as serve_http_module


@pytest.fixture(autouse=True)
def reset_http_singleton() -> None:
    """Reset HTTP singleton runtime between tests."""
    serve_http_module._HTTP_RUNTIME_SINGLETON = None


@pytest.mark.asyncio
async def test_serve_http_via_granian_builds_expected_command() -> None:
    fake_process = AsyncMock()
    fake_process.wait = AsyncMock(return_value=0)

    with patch(
        "everspring_mcp.http.serve_http.asyncio.create_subprocess_exec",
        return_value=fake_process,
    ) as mock_create_subprocess:
        exit_code = await serve_http_module.serve_http_via_granian(
            host="127.0.0.1",
            port=9000,
            workers=2,
            backlog=512,
            threads=4,
            log_level="INFO",
        )

    assert exit_code == 0
    assert mock_create_subprocess.call_count == 1
    command_args = mock_create_subprocess.call_args.args
    assert "--workers" in command_args
    assert "2" in command_args
    assert "--backlog" in command_args
    assert "512" in command_args
    assert "--runtime-threads" in command_args
    assert "4" in command_args
    assert "--host" in command_args
    assert "127.0.0.1" in command_args
    assert "--port" in command_args
    assert "9000" in command_args
    assert "--log-level" in command_args
    assert "info" in command_args
    assert "everspring_mcp.http.serve_http:create_http_app" in command_args
    assert fake_process.wait.await_count == 1


@pytest.mark.asyncio
async def test_serve_http_via_granian_uses_env_defaults() -> None:
    fake_process = AsyncMock()
    fake_process.wait = AsyncMock(return_value=0)

    with patch.dict(
        "os.environ",
        {
            "EVERSPRING_HTTP_HOST": "0.0.0.0",
            "EVERSPRING_HTTP_PORT": "7777",
        },
        clear=False,
    ):
        with patch(
            "everspring_mcp.http.serve_http.asyncio.create_subprocess_exec",
            return_value=fake_process,
        ) as mock_create_subprocess:
            exit_code = await serve_http_module.serve_http_via_granian()

    assert exit_code == 0
    command_args = mock_create_subprocess.call_args.args
    assert "0.0.0.0" in command_args
    assert "7777" in command_args


def test_http_runtime_exposes_sse_and_message_path_aliases() -> None:
    runtime = serve_http_module._HttpTransportRuntime()
    paths = {route.path for route in runtime.app.router.routes if hasattr(route, "path")}

    assert "/sse" in paths
    assert "/sse/" in paths
    assert "/sse/messages" in paths
    assert "/messages" in paths


def test_http_post_returns_409_for_closed_session_writer() -> None:
    runtime = serve_http_module._HttpTransportRuntime()
    session_id = uuid4()

    class _ClosedWriter:
        async def send(self, _item: object) -> None:
            raise ClosedResourceError

    runtime._transport._read_stream_writers[session_id] = _ClosedWriter()  # noqa: SLF001

    with TestClient(runtime.app) as client:
        response = client.post(
            f"/messages?session_id={session_id.hex}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            },
        )

    assert response.status_code == 409
    assert "Reconnect to /sse" in response.text
    assert session_id not in runtime._transport._read_stream_writers  # noqa: SLF001


def test_http_post_accepts_valid_message_for_open_session_writer() -> None:
    runtime = serve_http_module._HttpTransportRuntime()
    session_id = uuid4()
    sent: list[object] = []

    class _OpenWriter:
        async def send(self, item: object) -> None:
            sent.append(item)

    runtime._transport._read_stream_writers[session_id] = _OpenWriter()  # noqa: SLF001

    with TestClient(runtime.app) as client:
        response = client.post(
            f"/sse/messages?session_id={session_id.hex}",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            },
        )

    assert response.status_code == 202
    assert len(sent) == 1
