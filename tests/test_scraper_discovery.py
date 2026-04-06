"""Tests for discovery URL filtering rules."""

from __future__ import annotations

from everspring_mcp.models.content import ContentType
from everspring_mcp.scraper.discovery import should_skip_url


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
