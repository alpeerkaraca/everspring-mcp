"""EverSpring MCP - Pytest fixtures and configuration.

This module provides common test fixtures:
- Mock S3 bucket using moto
- Sample Spring documentation HTML
- SpringVersion and SpringModule fixtures
- Mock browser fixture
- Parser and browser configuration fixtures
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import boto3
import pytest
from moto import mock_aws

from everspring_mcp.models.content import ContentType
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.browser import BrowserConfig, SpringBrowser
from everspring_mcp.scraper.parser import ParserConfig, SpringDocParser
from everspring_mcp.scraper.pipeline import PipelineConfig, ScrapeTarget

# =============================================================================
# Sample HTML Fixtures
# =============================================================================


SAMPLE_SPRING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta property="og:title" content="Spring Boot Reference Documentation - Exämple">
    <title>Spring Boot Reference Documentation - Exämple</title>
</head>
<body>
    <span class="version">4.0.5</span>
    <nav class="toc">
        <ul>
            <li>
                <a href="/spring-boot/reference/getting-started/">Getting Started</a>
                <ul>
                    <li><a href="/spring-boot/reference/getting-started/first-app/">Your First Application</a></li>
                    <li><a href="/spring-boot/reference/getting-started/installing/">Installing Spring Boot</a></li>
                </ul>
            </li>
            <li>
                <a href="/spring-boot/reference/configuration/">Configuration</a>
                <ul>
                    <li><a href="/spring-boot/reference/configuration/properties/">Properties</a></li>
                    <li><a href="/spring-boot/reference/configuration/yaml/">YAML Support</a></li>
                </ul>
            </li>
            <li>
                <a href="/spring-boot/reference/actuator/">Actuator</a>
            </li>
        </ul>
    </nav>
    
    <article class="doc">
        <h1>Spring Boot Reference Documentation - Exämple</h1>
        
        <p>Welcome to the Spring Boot 4.0 reference documentation. This guide 
        provides comprehensive information about using Spring Boot.</p>
        
        <h2 id="getting-started">Getting Started</h2>
        
        <p>To get started with Spring Boot, create a new application:</p>
        
        <pre><code class="language-java">
package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
        </code></pre>
        
        <h3 id="using-annotations">Using Annotations</h3>
        
        <p>Spring Boot uses various annotations for configuration:</p>
        
        <pre><code class="language-java">
@RestController
@RequestMapping("/api")
public class MyController {
    
    @GetMapping("/hello")
    public String hello() {
        return "Hello, Spring Boot!";
    }
}
        </code></pre>
        
        <h2 id="configuration">Configuration</h2>
        
        <p>Configure your application using properties or YAML:</p>
        
        <pre><code class="language-yaml">
spring:
  application:
    name: my-app
  datasource:
    url: jdbc:h2:mem:testdb
    driver-class-name: org.h2.Driver
        </code></pre>
        
        <pre><code class="language-properties">
server.port=8080
spring.profiles.active=dev
        </code></pre>
        
        <h2 id="kotlin-support">Kotlin Support</h2>
        
        <p>Spring Boot fully supports Kotlin:</p>
        
        <pre><code class="language-kotlin">
@SpringBootApplication
class MyApplication

fun main(args: Array<String>) {
    runApplication<MyApplication>(*args)
}
        </code></pre>
    </article>
</body>
</html>
"""


MINIMAL_SPRING_HTML = """<!DOCTYPE html>
<html>
<head><title>Minimal Page</title></head>
<body>
    <span class="version">4.0.5</span>
    <main>
        <h1>Minimal Content</h1>
        <p>This is minimal content for testing.</p>
    </main>
</body>
</html>
"""


EMPTY_SIDEBAR_HTML = """<!DOCTYPE html>
<html>
<head><title>No Sidebar</title></head>
<body>
    <span class="version">4.0.5</span>
    <article class="doc">
        <h1>Page Without Sidebar</h1>
        <p>This page has no navigation sidebar.</p>
    </article>
</body>
</html>
"""


# =============================================================================
# HTML Fixtures
# =============================================================================


@pytest.fixture
def sample_spring_html() -> str:
    """Sample Spring Boot documentation HTML."""
    return SAMPLE_SPRING_HTML


@pytest.fixture
def minimal_spring_html() -> str:
    """Minimal HTML for basic parsing tests."""
    return MINIMAL_SPRING_HTML


@pytest.fixture
def empty_sidebar_html() -> str:
    """HTML without sidebar navigation."""
    return EMPTY_SIDEBAR_HTML


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def spring_boot_module() -> SpringModule:
    """Spring Boot module enum."""
    return SpringModule.BOOT


@pytest.fixture
def spring_framework_module() -> SpringModule:
    """Spring Framework module enum."""
    return SpringModule.FRAMEWORK


@pytest.fixture
def spring_boot_version() -> SpringVersion:
    """Valid Spring Boot 4.0.5 version."""
    return SpringVersion(
        module=SpringModule.BOOT,
        major=4,
        minor=0,
        patch=5,
    )


@pytest.fixture
def spring_framework_version() -> SpringVersion:
    """Valid Spring Framework 7.0.0 version."""
    return SpringVersion(
        module=SpringModule.FRAMEWORK,
        major=7,
        minor=0,
        patch=0,
    )


@pytest.fixture
def scrape_target(spring_boot_module: SpringModule) -> ScrapeTarget:
    """Sample scrape target for Spring Boot."""
    return ScrapeTarget(
        url="https://docs.spring.io/spring-boot/reference/",
        module=spring_boot_module,
        major=4,
        minor=0,
        patch=0,
        content_type=ContentType.REFERENCE,
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def browser_config() -> BrowserConfig:
    """Browser configuration for testing."""
    return BrowserConfig(
        headless=True,
        timeout_ms=10000,
        max_retries=2,
    )


@pytest.fixture
def parser_config() -> ParserConfig:
    """Parser configuration for testing."""
    return ParserConfig()


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Pipeline configuration for testing with mock S3."""
    return PipelineConfig(
        s3_bucket="test-bucket",
        s3_prefix="test-docs",
        aws_region="us-east-1",
        enable_hash_check=True,
    )


# =============================================================================
# Parser Fixtures
# =============================================================================


@pytest.fixture
def parser(parser_config: ParserConfig) -> SpringDocParser:
    """Configured SpringDocParser instance."""
    return SpringDocParser(parser_config)


# =============================================================================
# AWS/S3 Mock Fixtures
# =============================================================================


@pytest.fixture
def aws_credentials() -> None:
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def mock_s3(aws_credentials: None) -> Generator[Any, None, None]:
    """Mock S3 client with test bucket using moto."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield s3


@pytest.fixture
def mock_s3_with_content(mock_s3: Any) -> Any:
    """Mock S3 with pre-existing content for hash checking tests."""
    content = "# Existing Content\n\nThis is existing content."
    content_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    mock_s3.put_object(
        Bucket="test-bucket",
        Key="test-docs/spring-boot/4.0.0/existing.md",
        Body=content.encode("utf-8"),
        ContentType="text/markdown",
        Metadata={
            "content-hash": content_hash,
            "schema-version": "1",
        },
    )
    return mock_s3


# =============================================================================
# Browser Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_browser_response(sample_spring_html: str) -> AsyncMock:
    """Mock Playwright Response object."""
    response = AsyncMock()
    response.status = 200
    response.status_text = "OK"
    response.headers = {}
    return response


@pytest.fixture
def mock_browser(
    sample_spring_html: str,
    mock_browser_response: AsyncMock,
) -> AsyncMock:
    """Mock SpringBrowser that returns sample HTML."""
    browser = AsyncMock(spec=SpringBrowser)

    # Setup context manager
    browser.__aenter__ = AsyncMock(return_value=browser)
    browser.__aexit__ = AsyncMock(return_value=None)

    # Setup methods
    browser.navigate = AsyncMock(return_value=mock_browser_response)
    browser.navigate_with_retry = AsyncMock(return_value=mock_browser_response)
    browser.fast_precheck = AsyncMock(return_value=None)
    browser.last_precheck_hash = None
    browser.get_html = AsyncMock(return_value=sample_spring_html)
    browser.get_html_with_fallback = AsyncMock(return_value=sample_spring_html)
    browser.get_content = AsyncMock(return_value="Spring Boot Reference Documentation")
    browser.is_launched = True
    browser.current_url = "https://docs.spring.io/spring-boot/reference/"

    return browser


@pytest.fixture
def mock_browser_factory(mock_browser: AsyncMock) -> MagicMock:
    """Factory that returns the mock browser."""
    factory = MagicMock()
    factory.return_value = mock_browser
    return factory


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def pipeline_env_vars() -> Generator[None, None, None]:
    """Set environment variables for pipeline configuration."""
    original_env = os.environ.copy()

    os.environ["EVERSPRING_S3_BUCKET"] = "test-bucket"
    os.environ["EVERSPRING_S3_PREFIX"] = "test-docs"
    os.environ["AWS_REGION"] = "us-east-1"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def sample_urls() -> list[str]:
    """Sample URLs for discovery tests."""
    return [
        "https://docs.spring.io/spring-boot/reference/",
        "https://docs.spring.io/spring-boot/reference/getting-started/",
        "https://docs.spring.io/spring-boot/reference/configuration/",
        "https://docs.spring.io/spring-boot/reference/actuator/",
    ]


@pytest.fixture
def external_urls() -> list[str]:
    """External URLs that should be filtered out."""
    return [
        "https://github.com/spring-projects/spring-boot",
        "https://stackoverflow.com/questions/tagged/spring-boot",
        "mailto:support@spring.io",
    ]
