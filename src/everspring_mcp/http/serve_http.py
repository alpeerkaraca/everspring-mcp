"""HTTP transport runner for MCP server (SSE + POST messages)."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
from uuid import UUID

import mcp.types as mcp_types
from anyio import BrokenResourceError, ClosedResourceError
from fastapi import FastAPI, Request
from mcp.server import NotificationOptions, Server
from mcp.server.sse import SseServerTransport
from mcp.shared.message import ServerMessageMetadata, SessionMessage
from pydantic import ValidationError
from starlette.responses import Response

from everspring_mcp.utils.logging import get_logger

logger = get_logger("http.serve_http")

DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8000


class _HttpTransportRuntime:
    """Singleton HTTP runtime that owns FastAPI app and MCP transport wiring."""

    def __init__(self) -> None:
        # Mounted SSE app runs under /sse, so MCP should post to /sse/messages.
        self._transport = SseServerTransport("/sse/messages")
        self._mcp_server: Server[Any, Any] | None = None
        self._server_name = "everspring-mcp"
        self.app = FastAPI(title=f"{self._server_name} MCP HTTP Server")
        self._mount_routes()

    def bind(self, mcp_server: Server[Any, Any], *, server_name: str) -> None:
        self._mcp_server = mcp_server
        self._server_name = server_name
        self.app.title = f"{server_name} MCP HTTP Server"

    def _require_mcp_server(self) -> Server[Any, Any]:
        if self._mcp_server is None:
            raise RuntimeError("HTTP runtime is not bound to an MCP server")
        return self._mcp_server

    def _mount_routes(self) -> None:
        @self.app.get("/healthz")
        async def _healthz() -> dict[str, str]:
            return {"status": "ok"}

        class _SSEEndpoint:
            def __init__(self, runtime: _HttpTransportRuntime) -> None:
                self._runtime = runtime

            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                mcp_server = self._runtime._require_mcp_server()
                try:
                    async with self._runtime._transport.connect_sse(
                        scope,
                        receive,
                        send,
                    ) as (
                        read_stream,
                        write_stream,
                    ):
                        await mcp_server.run(
                            read_stream,
                            write_stream,
                            mcp_server.create_initialization_options(
                                notification_options=NotificationOptions(),
                            ),
                        )
                except (ClosedResourceError, BrokenResourceError):
                    logger.debug("SSE client disconnected")
                except RuntimeError as exc:
                    # Granian can surface a late ASGI flow race when a client disconnects
                    # exactly as a response is being finalized.
                    if "Response already started" in str(exc):
                        logger.debug("Ignoring SSE response-finalization race: %s", exc)
                        return
                    raise

        sse_endpoint = _SSEEndpoint(self)
        self.app.router.add_route("/sse", sse_endpoint, methods=["GET"])
        self.app.router.add_route("/sse/", sse_endpoint, methods=["GET"])

        async def _handle_message_post(request: Request) -> Response:
            session_id_param = request.query_params.get("session_id")
            if session_id_param is None:
                return Response("session_id is required", status_code=400)

            try:
                session_id = UUID(hex=session_id_param)
            except ValueError:
                return Response("Invalid session ID", status_code=400)

            writer = self._transport._read_stream_writers.get(session_id)  # noqa: SLF001
            if writer is None:
                return Response("Could not find session", status_code=404)

            body = await request.body()
            try:
                message = mcp_types.JSONRPCMessage.model_validate_json(body)
            except ValidationError:
                return Response("Could not parse message", status_code=400)

            session_message = SessionMessage(
                message,
                metadata=ServerMessageMetadata(request_context=request),
            )
            try:
                await writer.send(session_message)
            except (ClosedResourceError, BrokenResourceError):
                self._transport._read_stream_writers.pop(session_id, None)  # noqa: SLF001
                return Response(
                    "Session is closed. Reconnect to /sse and retry the request.",
                    status_code=409,
                )

            return Response("Accepted", status_code=202)

        # Canonical endpoint emitted by SseServerTransport("/messages").
        self.app.add_api_route("/messages", _handle_message_post, methods=["POST"])
        self.app.add_api_route("/messages/", _handle_message_post, methods=["POST"])
        # Backward-compatible alias for previous mounted /sse path behavior.
        self.app.add_api_route("/sse/messages", _handle_message_post, methods=["POST"])
        self.app.add_api_route("/sse/messages/", _handle_message_post, methods=["POST"])

_HTTP_RUNTIME_SINGLETON: _HttpTransportRuntime | None = None


def _get_http_runtime() -> _HttpTransportRuntime:
    global _HTTP_RUNTIME_SINGLETON
    if _HTTP_RUNTIME_SINGLETON is None:
        _HTTP_RUNTIME_SINGLETON = _HttpTransportRuntime()
    return _HTTP_RUNTIME_SINGLETON


def _resolve_http_host() -> str:
    return os.environ.get("EVERSPRING_HTTP_HOST", DEFAULT_HTTP_HOST)


def _resolve_http_port() -> int:
    raw_port = os.environ.get("EVERSPRING_HTTP_PORT")
    if raw_port is None:
        return DEFAULT_HTTP_PORT
    try:
        return int(raw_port)
    except ValueError:
        logger.warning(
            "Invalid EVERSPRING_HTTP_PORT value '%s'; falling back to %s",
            raw_port,
            DEFAULT_HTTP_PORT,
        )
        return DEFAULT_HTTP_PORT


async def serve_http(
    mcp_server: Server[Any, Any],
    *,
    server_name: str,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Backward-compatible helper for in-process HTTP wiring."""
    runtime = _get_http_runtime()
    runtime.bind(mcp_server, server_name=server_name)


def create_http_app() -> FastAPI:
    """Build the ASGI app for Granian's non-embedded factory mode."""
    from everspring_mcp.mcp.server import create_server
    from everspring_mcp.vector.config import VectorConfig

    runtime = _get_http_runtime()
    mcp_server = create_server(config=VectorConfig.from_env())
    runtime.bind(mcp_server.mcp, server_name=mcp_server.name)
    runtime.app.state.mcp_server = mcp_server

    if not getattr(runtime.app.state, "_mcp_startup_hook_registered", False):

        @runtime.app.on_event("startup")
        async def _initialize_mcp_server() -> None:
            await runtime.app.state.mcp_server.initialize()

        runtime.app.state._mcp_startup_hook_registered = True

    return runtime.app


async def serve_http_via_granian(
    *,
    host: str | None = None,
    port: int | None = None,
    workers: int | None = None,
    backlog: int | None = None,
    threads: int | None = None,
    log_level: str = "info",
    env: dict[str, str] | None = None,
) -> int:
    """Launch HTTP transport with Granian standard server (non-embedded)."""
    resolved_host = host or _resolve_http_host()
    resolved_port = port if port is not None else _resolve_http_port()
    command = [
        sys.executable,
        "-m",
        "granian",
        "--interface",
        "asgi",
        "--factory",
        "--host",
        resolved_host,
        "--port",
        str(resolved_port),
        "--log-level",
        log_level.lower(),
    ]
    if workers is not None:
        command.extend(["--workers", str(workers)])
    if backlog is not None:
        command.extend(["--backlog", str(backlog)])
    if threads is not None:
        command.extend(["--runtime-threads", str(threads)])
    command.append("everspring_mcp.http.serve_http:create_http_app")

    logger.info(
        "Starting HTTP MCP transport via Granian at http://%s:%s (SSE: /sse, POST: /messages; alias: /sse/messages)",
        resolved_host,
        resolved_port,
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        env=env,
    )
    return await process.wait()
