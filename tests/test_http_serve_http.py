"""Tests for singleton HTTP transport runtime."""

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
async def test_serve_http_uses_singleton_granian_server() -> None:
    fake_mcp_server = AsyncMock()
    fake_granian_server = AsyncMock()
    fake_granian_server.serve = AsyncMock(return_value=None)

    with patch(
        "everspring_mcp.http.serve_http.GranianEmbeddedServer",
        return_value=fake_granian_server,
    ) as mock_granian_server:
        await serve_http_module.serve_http(
            fake_mcp_server,
            server_name="singleton-test",
            host="127.0.0.1",
            port=9000,
        )
        await serve_http_module.serve_http(
            fake_mcp_server,
            server_name="singleton-test",
            host="127.0.0.1",
            port=9000,
        )

    assert mock_granian_server.call_count == 1
    assert fake_granian_server.serve.await_count == 2


@pytest.mark.asyncio
async def test_serve_http_rejects_host_port_change_after_singleton_init() -> None:
    fake_mcp_server = AsyncMock()
    fake_granian_server = AsyncMock()
    fake_granian_server.serve = AsyncMock(return_value=None)

    with patch(
        "everspring_mcp.http.serve_http.GranianEmbeddedServer",
        return_value=fake_granian_server,
    ):
        await serve_http_module.serve_http(
            fake_mcp_server,
            server_name="singleton-test",
            host="127.0.0.1",
            port=9000,
        )

        with pytest.raises(RuntimeError, match="different host/port"):
            await serve_http_module.serve_http(
                fake_mcp_server,
                server_name="singleton-test",
                host="127.0.0.1",
                port=9001,
            )


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
