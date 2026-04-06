"""EverSpring MCP - Browser fast-precheck tests."""

from __future__ import annotations

from typing import Any

import pytest

from everspring_mcp.scraper.browser import BrowserConfig, NotModifiedSignal, SpringBrowser
from everspring_mcp.scraper.exceptions import NavigationError


class _FakeHttpxResponse:
    """Minimal httpx response stub for fast-precheck tests."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    """Minimal async client stub compatible with `async with httpx.AsyncClient`."""

    def __init__(self, response: _FakeHttpxResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        return None

    async def get(
        self, url: str, headers: dict[str, str] | None = None
    ) -> _FakeHttpxResponse:
        assert url.startswith("https://")
        assert headers is not None
        return self._response


class _FakeNavigateResponse:
    """Minimal response object for navigate_with_retry tests."""

    status = 200


@pytest.mark.asyncio
async def test_fast_precheck_returns_not_modified_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast precheck should return NotModifiedSignal when hash matches stored hash."""
    html = """
    <html><body><main><h1>Spring Boot</h1><p>Reference docs content</p></main></body></html>
    """
    expected_hash = SpringBrowser.compute_core_content_hash(html)
    assert expected_hash is not None

    def _client_factory(*args: Any, **kwargs: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(_FakeHttpxResponse(200, html))

    monkeypatch.setattr(
        "everspring_mcp.scraper.browser.httpx.AsyncClient", _client_factory
    )

    browser = SpringBrowser()
    signal = await browser.fast_precheck(
        "https://docs.spring.io/spring-boot/reference/",
        stored_hash=expected_hash,
    )

    assert isinstance(signal, NotModifiedSignal)
    assert signal.signal == "NotModified"
    assert signal.content_hash == expected_hash
    assert browser.last_precheck_hash == expected_hash


@pytest.mark.asyncio
async def test_fast_precheck_sets_hash_when_content_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast precheck should not signal NotModified when hashes differ."""
    html = """
    <html><body><div class="doc">Updated reference content</div></body></html>
    """
    expected_hash = SpringBrowser.compute_core_content_hash(html)
    assert expected_hash is not None

    def _client_factory(*args: Any, **kwargs: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(_FakeHttpxResponse(200, html))

    monkeypatch.setattr(
        "everspring_mcp.scraper.browser.httpx.AsyncClient", _client_factory
    )

    browser = SpringBrowser()
    signal = await browser.fast_precheck(
        "https://docs.spring.io/spring-framework/reference/",
        stored_hash="0" * 64,
    )

    assert signal is None
    assert browser.last_precheck_hash == expected_hash


@pytest.mark.asyncio
async def test_fast_precheck_gracefully_skips_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast precheck should return None for non-success HTTP responses."""

    def _client_factory(*args: Any, **kwargs: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(
            _FakeHttpxResponse(503, "<html><body>Unavailable</body></html>")
        )

    monkeypatch.setattr(
        "everspring_mcp.scraper.browser.httpx.AsyncClient", _client_factory
    )

    browser = SpringBrowser()
    signal = await browser.fast_precheck(
        "https://docs.spring.io/spring-boot/reference/", stored_hash="a" * 64
    )

    assert signal is None
    assert browser.last_precheck_hash is None


@pytest.mark.asyncio
async def test_navigate_with_retry_retries_transient_navigation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient DNS failures should be retried with context rotation."""
    browser = SpringBrowser(BrowserConfig(max_retries=3, base_retry_delay=0.5))
    attempts = {"navigate": 0, "rotate": 0}
    sleep_calls: list[float] = []

    async def _fake_navigate(url: str) -> _FakeNavigateResponse:
        attempts["navigate"] += 1
        if attempts["navigate"] < 3:
            raise NavigationError(
                "Navigation failed: Page.goto: net::ERR_NAME_NOT_RESOLVED",
                url=url,
            )
        return _FakeNavigateResponse()

    async def _fake_rotate() -> None:
        attempts["rotate"] += 1

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(browser, "navigate", _fake_navigate)
    monkeypatch.setattr(browser, "_rotate_context", _fake_rotate)
    monkeypatch.setattr("everspring_mcp.scraper.browser.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("everspring_mcp.scraper.browser.random.uniform", lambda _a, _b: 0.0)

    response = await browser.navigate_with_retry("https://docs.spring.io/spring-framework/reference/")

    assert isinstance(response, _FakeNavigateResponse)
    assert attempts["navigate"] == 3
    assert attempts["rotate"] == 2
    assert sleep_calls == [0.5, 1.0]


@pytest.mark.asyncio
async def test_navigate_with_retry_does_not_retry_non_transient_navigation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 404 should surface immediately without transient retry loop."""
    browser = SpringBrowser(BrowserConfig(max_retries=4, base_retry_delay=0.5))
    attempts = {"navigate": 0, "rotate": 0}

    async def _fake_navigate(url: str) -> _FakeNavigateResponse:
        attempts["navigate"] += 1
        raise NavigationError("HTTP 404: Not Found", url=url, status_code=404)

    async def _fake_rotate() -> None:
        attempts["rotate"] += 1

    monkeypatch.setattr(browser, "navigate", _fake_navigate)
    monkeypatch.setattr(browser, "_rotate_context", _fake_rotate)

    with pytest.raises(NavigationError, match="HTTP 404"):
        await browser.navigate_with_retry("https://docs.spring.io/spring-data/keyvalue/docs/current/api/missing.html")

    assert attempts["navigate"] == 1
    assert attempts["rotate"] == 0
