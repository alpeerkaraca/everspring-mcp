"""HTTP transport runner for MCP server (SSE + POST messages)."""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from granian.log import LogLevels
import mcp.types as mcp_types
from granian.constants import Interfaces
from granian.server.embed import Server as GranianEmbeddedServer
from granian.utils.proxies import wrap_asgi_with_proxy_headers
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
    """Singleton HTTP runtime that owns FastAPI app and granian server."""

    def __init__(self) -> None:
        # Mounted SSE app runs under /sse, so MCP should post to /sse/messages.
        self._transport = SseServerTransport("/sse/messages")
        self._mcp_server: Server[Any, Any] | None = None
        self._server_name = "everspring-mcp"
        self._bound_host: str | None = None
        self._bound_port: int | None = None
        self._granian_server: GranianEmbeddedServer | None = None
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

        async def _handle_sse(request: Request) -> Response:
            mcp_server = self._require_mcp_server()
            async with self._transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
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
            return Response()

        self.app.add_api_route("/sse", _handle_sse, methods=["GET"])
        self.app.add_api_route("/sse/", _handle_sse, methods=["GET"])

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

    async def serve(self, *, host: str, port: int) -> None:
        if self._mcp_server is None:
            raise RuntimeError("HTTP runtime cannot start before MCP server bind")

        if self._granian_server is None:
            self._bound_host = host
            self._bound_port = port

            proxy_headers = True
            forwarded_allow_ips = ""
            asgi_app = wrap_asgi_with_proxy_headers(self.app, "*")
            self._granian_server = GranianEmbeddedServer(
                target=asgi_app,
                address=host,
                port=port,
                interface=Interfaces.ASGI,
                log_level=LogLevels.info,
            )
        elif self._bound_host != host or self._bound_port != port:
            raise RuntimeError(
                "HTTP singleton already initialized with a different host/port"
            )

        await self._granian_server.serve()


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
    """Serve MCP over HTTP using SSE transport."""
    resolved_host = host or _resolve_http_host()
    resolved_port = port if port is not None else _resolve_http_port()
    runtime = _get_http_runtime()
    runtime.bind(mcp_server, server_name=server_name)

    logger.info(
        "Starting HTTP MCP transport for %s at http://%s:%s (SSE: /sse, POST: /messages; alias: /sse/messages)",
        server_name,
        resolved_host,
        resolved_port,
    )
    await runtime.serve(host=resolved_host, port=resolved_port)
