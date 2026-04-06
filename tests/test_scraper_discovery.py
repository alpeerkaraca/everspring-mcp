"""Tests for discovery URL filtering rules."""

from __future__ import annotations

import pytest

from everspring_mcp.models.content import ContentType
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.discovery import DiscoveryConfig, SpringDocDiscovery, should_skip_url
from everspring_mcp.scraper.exceptions import NavigationError


def test_should_skip_url_reference_skips_api_and_javadoc() -> None:
    skip_ext = {".png"}
    assert should_skip_url(
        "https://docs.spring.io/spring-boot/4.0.5/api/",
        skip_ext,
        ContentType.REFERENCE,
    )
    assert should_skip_url(
        "https://docs.spring.io/spring-framework/docs/current/javadoc-api/",
        skip_ext,
        ContentType.REFERENCE,
    )


def test_should_skip_url_api_doc_allows_api_and_javadoc() -> None:
    skip_ext = {".png"}
    assert not should_skip_url(
        "https://docs.spring.io/spring-boot/4.0.5/api/",
        skip_ext,
        ContentType.API_DOC,
    )
    assert not should_skip_url(
        "https://docs.spring.io/spring-framework/docs/current/javadoc-api/",
        skip_ext,
        ContentType.API_DOC,
    )


def test_should_skip_url_still_skips_global_patterns() -> None:
    skip_ext = {".png"}
    assert should_skip_url(
        "https://github.com/spring-projects/spring-boot",
        skip_ext,
        ContentType.API_DOC,
    )
    assert should_skip_url(
        "mailto:support@spring.io",
        skip_ext,
        ContentType.API_DOC,
    )


class _Always404Browser:
    """Browser stub that always returns HTTP 404 on navigation."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _Always404Browser:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        return None

    async def navigate_with_retry(self, url: str) -> None:
        raise NavigationError("HTTP 404: Not Found", url=url, status_code=404)

    async def get_html(self) -> str:
        return "<html></html>"


@pytest.mark.asyncio
async def test_discovery_suppresses_http_404_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Links returning 404 should be excluded when suppression is enabled."""
    monkeypatch.setattr("everspring_mcp.scraper.discovery.SpringBrowser", _Always404Browser)
    discovery = SpringDocDiscovery(DiscoveryConfig(suppress_http_404=True, max_depth=1))

    result = await discovery.discover(
        entry_url="https://docs.spring.io/spring-boot/reference/",
        module=SpringModule.BOOT,
        version=SpringVersion(module=SpringModule.BOOT, major=4, minor=0, patch=5),
        content_type=ContentType.REFERENCE,
    )

    assert result.link_count == 0


@pytest.mark.asyncio
async def test_discovery_keeps_http_404_links_when_suppression_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When suppression is disabled, discovered links are kept even if navigation returns 404."""
    monkeypatch.setattr("everspring_mcp.scraper.discovery.SpringBrowser", _Always404Browser)
    discovery = SpringDocDiscovery(DiscoveryConfig(suppress_http_404=False, max_depth=1))

    result = await discovery.discover(
        entry_url="https://docs.spring.io/spring-boot/reference/",
        module=SpringModule.BOOT,
        version=SpringVersion(module=SpringModule.BOOT, major=4, minor=0, patch=5),
        content_type=ContentType.REFERENCE,
    )

    assert result.link_count == 1
