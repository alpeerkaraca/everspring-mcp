"""EverSpring MCP - Parser and model validation tests.

Tests for:
- SpringVersion validation (v4+, v7+ rules)
- Invalid version rejection
- SpringDocParser with sample HTML
- ScrapedPage model with correct schema_version
- Code block extraction and language detection
- Sidebar navigation parsing
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everspring_mcp.models.base import VersionedModel, compute_hash
from everspring_mcp.models.content import (
    CodeLanguage,
    ContentType,
    DocumentSection,
    ScrapedPage,
)
from everspring_mcp.models.spring import SpringModule, SpringVersion, VersionRange
from everspring_mcp.scraper.exceptions import ContentExtractionError
from everspring_mcp.scraper.parser import (
    LANGUAGE_PATTERNS,
    SPRING_DOC_SELECTORS,
    SpringDocParser,
)

# =============================================================================
# SpringVersion Validation Tests
# =============================================================================


class TestSpringVersionValidation:
    """Tests for SpringVersion minimum version validation."""

    def test_valid_spring_boot_v4(self) -> None:
        """Spring Boot version 4+ should be valid."""
        version = SpringVersion(
            module=SpringModule.BOOT,
            major=4,
            minor=0,
            patch=0,
        )
        assert version.major == 4
        assert version.module == SpringModule.BOOT
        assert version.version_string == "4.0.0"

    def test_valid_spring_boot_v4_with_qualifier(self) -> None:
        """Spring Boot 4.0.0-RELEASE should be valid."""
        version = SpringVersion(
            module=SpringModule.BOOT,
            major=4,
            minor=1,
            patch=2,
            qualifier="RELEASE",
        )
        assert version.version_string == "4.1.2-RELEASE"

    def test_valid_spring_framework_v7(self) -> None:
        """Spring Framework version 7+ should be valid."""
        version = SpringVersion(
            module=SpringModule.FRAMEWORK,
            major=7,
            minor=0,
            patch=0,
        )
        assert version.major == 7
        assert version.module == SpringModule.FRAMEWORK

    def test_valid_spring_security_v6(self) -> None:
        """Spring Security version 6+ should be valid."""
        version = SpringVersion(
            module=SpringModule.SECURITY,
            major=6,
            minor=0,
            patch=0,
        )
        assert version.major == 6

    def test_valid_spring_data_v4(self) -> None:
        """Spring Data version 4+ should be valid."""
        version = SpringVersion(
            module=SpringModule.DATA,
            major=4,
            minor=0,
            patch=0,
        )
        assert version.major == 4

    def test_valid_spring_cloud_v4(self) -> None:
        """Spring Cloud version 4+ should be valid."""
        version = SpringVersion(
            module=SpringModule.CLOUD,
            major=4,
            minor=0,
            patch=0,
        )
        assert version.major == 4

    def test_invalid_spring_boot_v3(self) -> None:
        """Spring Boot version 3 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpringVersion(
                module=SpringModule.BOOT,
                major=3,
                minor=2,
                patch=0,
            )
        assert "requires version 4+" in str(exc_info.value)

    def test_invalid_spring_framework_v6(self) -> None:
        """Spring Framework version 6 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpringVersion(
                module=SpringModule.FRAMEWORK,
                major=6,
                minor=0,
                patch=0,
            )
        assert "requires version 7+" in str(exc_info.value)

    def test_invalid_spring_security_v5(self) -> None:
        """Spring Security version 5 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpringVersion(
                module=SpringModule.SECURITY,
                major=5,
                minor=0,
                patch=0,
            )
        assert "requires version 6+" in str(exc_info.value)

    def test_version_comparison(self) -> None:
        """Test version comparison operators."""
        v4_0_0 = SpringVersion(module=SpringModule.BOOT, major=4, minor=0, patch=0)
        v4_1_0 = SpringVersion(module=SpringModule.BOOT, major=4, minor=1, patch=0)
        v4_1_1 = SpringVersion(module=SpringModule.BOOT, major=4, minor=1, patch=1)

        assert v4_0_0 < v4_1_0
        assert v4_1_0 < v4_1_1
        assert v4_1_1 > v4_0_0
        assert v4_1_0 >= v4_0_0
        assert v4_0_0 <= v4_1_0
        assert v4_0_0 == v4_0_0

    def test_version_comparison_different_modules_raises(self) -> None:
        """Comparing versions of different modules should raise."""
        boot = SpringVersion(module=SpringModule.BOOT, major=4, minor=0, patch=0)
        framework = SpringVersion(module=SpringModule.FRAMEWORK, major=7, minor=0, patch=0)

        with pytest.raises(ValueError, match="different modules"):
            _ = boot < framework

    def test_version_parse(self) -> None:
        """Test parsing version strings."""
        version = SpringVersion.parse(SpringModule.BOOT, "4.1.2-RELEASE")
        assert version.major == 4
        assert version.minor == 1
        assert version.patch == 2
        assert version.qualifier == "RELEASE"

    def test_version_tuple(self) -> None:
        """Test version tuple property."""
        version = SpringVersion(module=SpringModule.BOOT, major=4, minor=1, patch=2)
        assert version.version_tuple == (4, 1, 2)

    def test_schema_version_present(self) -> None:
        """Test that schema_version is present in model."""
        version = SpringVersion(module=SpringModule.BOOT, major=4)
        assert version.schema_version == VersionedModel.CURRENT_SCHEMA_VERSION


class TestVersionRange:
    """Tests for VersionRange model."""

    def test_version_range_contains(self) -> None:
        """Test version range containment check."""
        min_ver = SpringVersion(module=SpringModule.BOOT, major=4, minor=0)
        max_ver = SpringVersion(module=SpringModule.BOOT, major=4, minor=5)

        range_ = VersionRange(
            module=SpringModule.BOOT,
            min_version=min_ver,
            max_version=max_ver,
        )

        v4_2 = SpringVersion(module=SpringModule.BOOT, major=4, minor=2)
        v4_6 = SpringVersion(module=SpringModule.BOOT, major=4, minor=6)

        assert range_.contains(v4_2) is True
        assert range_.contains(v4_6) is False

    def test_version_range_invalid_min_max(self) -> None:
        """Min version cannot be greater than max."""
        with pytest.raises(ValidationError):
            VersionRange(
                module=SpringModule.BOOT,
                min_version=SpringVersion(module=SpringModule.BOOT, major=4, minor=5),
                max_version=SpringVersion(module=SpringModule.BOOT, major=4, minor=0),
            )


# =============================================================================
# SpringDocParser Tests
# =============================================================================


class TestSpringDocParser:
    """Tests for SpringDocParser HTML parsing."""

    def test_parser_initialization(self, parser: SpringDocParser) -> None:
        """Test parser initializes correctly."""
        assert parser.config is not None

    def test_extract_title(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
    ) -> None:
        """Test title extraction from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(sample_spring_html, "lxml")
        title = parser.extract_title(soup)

        assert title == "Spring Boot Reference Documentation - Exämple"

    def test_extract_title_from_og_meta(self, parser: SpringDocParser) -> None:
        """Test title extraction from og:title meta tag."""
        from bs4 import BeautifulSoup

        html = '<html><head><meta property="og:title" content="OG Title"></head></html>'
        soup = BeautifulSoup(html, "lxml")
        title = parser.extract_title(soup)

        assert title == "OG Title"

    def test_extract_sidebar_navigation(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
    ) -> None:
        """Test sidebar navigation extraction."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(sample_spring_html, "lxml")
        sidebar = parser.extract_sidebar(soup)

        assert len(sidebar) > 0
        # Check first item
        assert sidebar[0]["title"] == "Getting Started"
        assert sidebar[0]["url"] == "/spring-boot/reference/getting-started/"
        # Check nested children
        assert len(sidebar[0]["children"]) == 2
        assert sidebar[0]["children"][0]["title"] == "Your First Application"

    def test_extract_empty_sidebar(
        self,
        parser: SpringDocParser,
        empty_sidebar_html: str,
    ) -> None:
        """Test handling of missing sidebar."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(empty_sidebar_html, "lxml")
        sidebar = parser.extract_sidebar(soup)

        assert sidebar == []

    def test_extract_code_blocks(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
    ) -> None:
        """Test code block extraction."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(sample_spring_html, "lxml")
        code_blocks = parser.extract_code_blocks(soup)

        assert len(code_blocks) >= 4  # Java, YAML, Properties, Kotlin examples

        # Check Java detection
        java_blocks = [b for b in code_blocks if b.language == CodeLanguage.JAVA]
        assert len(java_blocks) >= 2

        # Check annotation extraction
        annotations_found = []
        for block in java_blocks:
            annotations_found.extend(block.annotations)
        assert "SpringBootApplication" in annotations_found
        assert "RestController" in annotations_found

    def test_extract_yaml_code_block(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
    ) -> None:
        """Test YAML code block detection."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(sample_spring_html, "lxml")
        code_blocks = parser.extract_code_blocks(soup)

        yaml_blocks = [b for b in code_blocks if b.language == CodeLanguage.YAML]
        assert len(yaml_blocks) >= 1
        assert "spring:" in yaml_blocks[0].code

    def test_extract_kotlin_code_block(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
    ) -> None:
        """Test Kotlin code block detection."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(sample_spring_html, "lxml")
        code_blocks = parser.extract_code_blocks(soup)

        kotlin_blocks = [b for b in code_blocks if b.language == CodeLanguage.KOTLIN]
        assert len(kotlin_blocks) >= 1
        assert "fun main" in kotlin_blocks[0].code

    def test_to_markdown_conversion(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
    ) -> None:
        """Test HTML to Markdown conversion."""
        markdown = parser.to_markdown(sample_spring_html)

        assert "# Spring Boot Reference Documentation - Exämple" in markdown
        assert "## Getting Started" in markdown
        assert "@SpringBootApplication" in markdown

    def test_to_markdown_removes_copied_artifacts(
        self,
        parser: SpringDocParser,
    ) -> None:
        """Spring docs copy-button artifacts should be removed from markdown."""
        html = """
        <article>
            <h1>Sample</h1>
            <pre><code class="language-java">class Demo {}</code></pre>
            <p>Copied!</p>
        </article>
        """

        markdown = parser.to_markdown(html)

        assert "class Demo" in markdown
        assert "Copied!" not in markdown

    def test_parse_prunes_navigation_edit_and_pagination_noise(
        self,
        parser: SpringDocParser,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Parser should remove noisy chrome while keeping article and code."""
        html = """
        <html>
          <body>
            <span class="version">4.0.5</span>
            <nav>Global Navigation Noise</nav>
            <aside>Aside Noise</aside>
            <footer>Footer Noise</footer>
            <article class="doc">
              <h1>Core Article</h1>
              <div class="toc">Table of contents noise</div>
              <a class="edit-link" href="/edit">Edit this page</a>
              <div class="pagination"><a rel="next" href="/next">Next</a></div>
              <p>Important article body content.</p>
              <pre><code class="language-java">class Demo {}</code></pre>
            </article>
          </body>
        </html>
        """

        scraped_page = parser.parse(
            html=html,
            url="https://docs.spring.io/spring-boot/reference/noise-test/",
            module=SpringModule.BOOT,
            version=spring_boot_version,
        )

        assert "Important article body content." in scraped_page.markdown_content
        assert "class Demo" in scraped_page.markdown_content
        assert "Global Navigation Noise" not in scraped_page.markdown_content
        assert "Aside Noise" not in scraped_page.markdown_content
        assert "Footer Noise" not in scraped_page.markdown_content
        assert "Table of contents noise" not in scraped_page.markdown_content
        assert "Edit this page" not in scraped_page.markdown_content
        assert "Next" not in scraped_page.markdown_content

    def test_parse_full_page(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Test full page parsing returns ScrapedPage model."""
        scraped_page = parser.parse(
            html=sample_spring_html,
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            version=spring_boot_version,
            submodule="redis",
            content_type=ContentType.REFERENCE,
        )

        assert isinstance(scraped_page, ScrapedPage)
        assert scraped_page.title == "Spring Boot Reference Documentation - Ex-mple"
        assert scraped_page.module == SpringModule.BOOT
        assert scraped_page.submodule == "redis"
        assert scraped_page.version == spring_boot_version
        assert scraped_page.content_type == ContentType.REFERENCE
        assert scraped_page.schema_version == VersionedModel.CURRENT_SCHEMA_VERSION

    def test_scraped_page_content_hash(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Test ScrapedPage has valid SHA-256 content hash."""
        scraped_page = parser.parse(
            html=sample_spring_html,
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            version=spring_boot_version,
        )

        # Content hash should be 64 hex characters
        assert len(scraped_page.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in scraped_page.content_hash)

        # Hash should match computed hash of markdown content
        expected_hash = compute_hash(scraped_page.markdown_content)
        assert scraped_page.content_hash == expected_hash

    def test_scraped_page_sections(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Test ScrapedPage sections are parsed correctly."""
        scraped_page = parser.parse(
            html=sample_spring_html,
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            version=spring_boot_version,
        )

        assert len(scraped_page.sections) > 0

        # Check section structure
        for section in scraped_page.sections:
            assert isinstance(section, DocumentSection)
            assert section.id is not None
            assert section.title is not None
            assert section.level >= 1

    def test_scraped_page_normalizes_heading_ids_with_dots(
        self,
        parser: SpringDocParser,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Heading ids with dots should be normalized to valid section ids."""
        html = """
        <html>
          <body>
            <span class="version">4.0.5</span>
            <article class="doc">
              <h2 id="using.build-systems.maven">Maven</h2>
              <p>Build with Maven.</p>
            </article>
          </body>
        </html>
        """
        scraped_page = parser.parse(
            html=html,
            url="https://docs.spring.io/spring-boot/reference/using/build-systems.html",
            module=SpringModule.BOOT,
            version=spring_boot_version,
            content_type=ContentType.REFERENCE,
        )

        assert len(scraped_page.sections) == 1
        assert scraped_page.sections[0].id == "using-build-systems-maven"

    def test_parse_minimal_html(
        self,
        parser: SpringDocParser,
        minimal_spring_html: str,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Test parsing minimal HTML content."""
        scraped_page = parser.parse(
            html=minimal_spring_html,
            url="https://docs.spring.io/spring-boot/reference/minimal/",
            module=SpringModule.BOOT,
            version=spring_boot_version,
        )

        assert scraped_page.title == "Minimal Content"
        assert "minimal content" in scraped_page.markdown_content.lower()

    def test_extract_version(self, parser: SpringDocParser, sample_spring_html: str) -> None:
        """Test version extraction from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(sample_spring_html, "lxml")
        version = parser.extract_version(soup)

        assert version == "4.0.5"

    def test_missing_version_raises(self, parser: SpringDocParser) -> None:
        """Test version extraction raises when missing."""
        from bs4 import BeautifulSoup

        html = "<html><body><h1>No version</h1></body></html>"
        soup = BeautifulSoup(html, "lxml")

        with pytest.raises(ContentExtractionError):
            parser.extract_version(soup)

    def test_version_mismatch_raises(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
        spring_framework_version: SpringVersion,
    ) -> None:
        """Test parse fails when version doesn't match expected."""
        with pytest.raises(ContentExtractionError):
            parser.parse(
                html=sample_spring_html,
                url="https://docs.spring.io/spring-boot/reference/",
                module=SpringModule.BOOT,
                version=spring_framework_version,
            )

    def test_reference_version_marker_inside_noise_container_is_preserved(
        self,
        parser: SpringDocParser,
    ) -> None:
        """Version marker must survive cleaning even when nested in noisy containers."""
        html = """
        <html>
          <body>
            <nav>
              <span class="version">4.0.5</span>
              <a href="/other">Navigation Link</a>
            </nav>
            <article class="doc">
              <h1>Reference Page</h1>
              <p>Main content body.</p>
            </article>
          </body>
        </html>
        """
        expected_version = SpringVersion(module=SpringModule.BOOT, major=4, minor=0, patch=4)
        with pytest.raises(ContentExtractionError, match="Version mismatch"):
            parser.parse(
                html=html,
                url="https://docs.spring.io/spring-boot/reference/reference-page.html",
                module=SpringModule.BOOT,
                version=expected_version,
                content_type=ContentType.REFERENCE,
            )

    def test_api_doc_uses_provided_version_when_page_lacks_version(
        self,
        parser: SpringDocParser,
        spring_boot_version: SpringVersion,
    ) -> None:
        """API docs should use provided release version when marker is missing."""
        html = """
        <html>
          <head><title>API Page</title></head>
          <body>
            <main>
              <h1>Package Summary</h1>
              <p>API docs usually don't render release version marker.</p>
            </main>
          </body>
        </html>
        """
        page = parser.parse(
            html=html,
            url="https://docs.spring.io/spring-boot/api/java/org/example/package-summary.html",
            module=SpringModule.BOOT,
            version=spring_boot_version,
            content_type=ContentType.API_DOC,
        )
        assert page.version == spring_boot_version

    def test_reference_uses_provided_version_when_page_lacks_version(
        self,
        parser: SpringDocParser,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Reference docs should use provided release version when marker is missing."""
        html = """
        <html>
          <head><title>Reference Page</title></head>
          <body>
            <main>
              <h1>Getting Started</h1>
              <p>Version marker is omitted on this page.</p>
            </main>
          </body>
        </html>
        """
        page = parser.parse(
            html=html,
            url="https://docs.spring.io/spring-boot/reference/getting-started.html",
            module=SpringModule.BOOT,
            version=spring_boot_version,
            content_type=ContentType.REFERENCE,
        )
        assert page.version == spring_boot_version

    def test_api_doc_requires_explicit_version_when_missing_marker(
        self,
        parser: SpringDocParser,
    ) -> None:
        """API docs without marker must fail if explicit version is not provided."""
        html = """
        <html>
          <head><title>API Page</title></head>
          <body><main><h1>Package Summary</h1></main></body>
        </html>
        """
        with pytest.raises(ContentExtractionError, match="requires an explicit release version"):
            parser.parse(
                html=html,
                url="https://docs.spring.io/spring-boot/api/java/org/example/package-summary.html",
                module=SpringModule.BOOT,
                version=None,
                content_type=ContentType.API_DOC,
            )

    def test_extract_api_signature_from_javadoc_html(
        self,
        parser: SpringDocParser,
    ) -> None:
        """Extract class definition, implemented interfaces, and method tags from Javadoc."""
        html = """
        <html>
          <head><title>RestTemplate (Spring Framework API)</title></head>
          <body>
            <div class="header">
              <div class="sub-title">package org.springframework.web.client</div>
              <h1 class="title">Class RestTemplate</h1>
            </div>
            <section class="class-description">
              <div class="type-signature">
                public class RestTemplate extends InterceptingHttpAccessor
                implements RestOperations, InitializingBean
              </div>
            </section>
            <section class="method-details">
              <section class="detail" id="exchange">
                <h3>exchange</h3>
                <div class="member-signature">
                  public &lt;T&gt; ResponseEntity&lt;T&gt; exchange(String url, HttpMethod method)
                </div>
                <dl class="notes">
                  <dt>Parameters:</dt>
                  <dd><code>url</code> - the URL</dd>
                  <dd><code>method</code> - the HTTP method</dd>
                  <dt>Returns:</dt>
                  <dd>the response entity</dd>
                </dl>
              </section>
            </section>
            <div class="description">
              <p>This is long descriptive prose and should not be part of the signature output.</p>
            </div>
          </body>
        </html>
        """

        signature = parser.extract_api_signature(html)

        assert signature.package == "org.springframework.web.client"
        assert signature.name == "RestTemplate"
        assert signature.type == "class"
        assert signature.extends == "InterceptingHttpAccessor"
        assert signature.implements == ["RestOperations", "InitializingBean"]
        assert len(signature.methods) == 1

        method = signature.methods[0]
        assert method.name == "exchange"
        assert method.return_type == "ResponseEntity<T>"
        assert [param.name for param in method.parameters] == ["url", "method"]
        assert method.parameters[0].description == "the URL"
        assert method.parameters[1].description == "the HTTP method"
        assert method.return_description == "the response entity"

    def test_parse_api_doc_uses_compact_api_signature_markdown(
        self,
        parser: SpringDocParser,
        spring_boot_version: SpringVersion,
    ) -> None:
        """API doc parse should keep only high-signal signature data when available."""
        html = """
        <html>
          <head><title>RestTemplate (Spring Framework API)</title></head>
          <body>
            <div class="header">
              <div class="sub-title">package org.springframework.web.client</div>
              <h1 class="title">Class RestTemplate</h1>
            </div>
            <section class="class-description">
              <div class="type-signature">
                public class RestTemplate implements RestOperations
              </div>
            </section>
            <section class="method-details">
              <section class="detail" id="exchange">
                <div class="member-signature">
                  public ResponseEntity&lt;String&gt; exchange(String url)
                </div>
                <dl class="notes">
                  <dt>Parameters:</dt>
                  <dd><code>url</code> - target endpoint URL</dd>
                  <dt>Returns:</dt>
                  <dd>response payload</dd>
                </dl>
              </section>
            </section>
            <div class="description">
              <p>
                This verbose paragraph should be excluded to keep RAG context high-signal.
              </p>
            </div>
          </body>
        </html>
        """

        page = parser.parse(
            html=html,
            url="https://docs.spring.io/spring-boot/api/java/org/springframework/web/client/RestTemplate.html",
            module=SpringModule.BOOT,
            version=spring_boot_version,
            content_type=ContentType.API_DOC,
        )

        assert page.sections == []
        assert "## Type Definition" in page.markdown_content
        assert "## Implemented Interfaces" in page.markdown_content
        assert "## Method Signatures" in page.markdown_content
        assert "`@param url`" in page.markdown_content
        assert "`@return`" in page.markdown_content
        assert "verbose paragraph" not in page.markdown_content.lower()


# =============================================================================
# Language Detection Tests
# =============================================================================


class TestLanguageDetection:
    """Tests for code language detection patterns."""

    @pytest.mark.parametrize(
        "class_name,expected_lang",
        [
            ("language-java", CodeLanguage.JAVA),
            ("highlight-java", CodeLanguage.JAVA),
            ("lang-kotlin", CodeLanguage.KOTLIN),
            ("language-kt", CodeLanguage.KOTLIN),
            ("language-yaml", CodeLanguage.YAML),
            ("language-yml", CodeLanguage.YAML),
            ("language-properties", CodeLanguage.PROPERTIES),
            ("language-json", CodeLanguage.JSON),
            ("language-shell", CodeLanguage.SHELL),
            ("language-bash", CodeLanguage.SHELL),
            ("language-sql", CodeLanguage.SQL),
            ("language-xml", CodeLanguage.XML),
            ("language-groovy", CodeLanguage.GROOVY),
        ],
    )
    def test_language_pattern_matching(
        self,
        class_name: str,
        expected_lang: CodeLanguage,
    ) -> None:
        """Test language detection from class names."""
        import re

        # Extract the language part
        for prefix in ("language-", "highlight-", "lang-"):
            if class_name.startswith(prefix):
                lang_str = class_name[len(prefix):]
                break
        else:
            lang_str = class_name

        # Find matching pattern
        matched_lang = None
        for pattern, lang in LANGUAGE_PATTERNS.items():
            if re.search(pattern, lang_str, re.IGNORECASE):
                matched_lang = lang
                break

        assert matched_lang == expected_lang


# =============================================================================
# Selector Configuration Tests
# =============================================================================


class TestSelectorConfiguration:
    """Tests for CSS selector configuration."""

    def test_default_title_selectors(self) -> None:
        """Test default title selectors are configured."""
        assert "title" in SPRING_DOC_SELECTORS
        assert len(SPRING_DOC_SELECTORS["title"]) > 0

    def test_default_sidebar_selectors(self) -> None:
        """Test default sidebar selectors are configured."""
        assert "sidebar" in SPRING_DOC_SELECTORS
        assert "nav.toc" in SPRING_DOC_SELECTORS["sidebar"]

    def test_default_content_selectors(self) -> None:
        """Test default content selectors are configured."""
        assert "main_content" in SPRING_DOC_SELECTORS
        assert "article.doc" in SPRING_DOC_SELECTORS["main_content"]

    def test_default_code_selectors(self) -> None:
        """Test default code block selectors are configured."""
        assert "code_blocks" in SPRING_DOC_SELECTORS
        assert "pre code" in SPRING_DOC_SELECTORS["code_blocks"]


# =============================================================================
# Content Hash Verification Tests
# =============================================================================


class TestContentHashVerification:
    """Tests for SHA-256 content hash verification."""

    def test_compute_hash_string(self) -> None:
        """Test hash computation from string."""
        content = "Hello, World!"
        hash_value = compute_hash(content)

        assert len(hash_value) == 64
        # Known SHA-256 of "Hello, World!"
        assert hash_value == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

    def test_compute_hash_bytes(self) -> None:
        """Test hash computation from bytes."""
        content = b"Hello, World!"
        hash_value = compute_hash(content)

        assert len(hash_value) == 64
        assert hash_value == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

    def test_hash_consistency(self) -> None:
        """Test hash is consistent for same content."""
        content = "Test content for hashing"

        hash1 = compute_hash(content)
        hash2 = compute_hash(content)

        assert hash1 == hash2

    def test_hash_different_for_different_content(self) -> None:
        """Test hash differs for different content."""
        hash1 = compute_hash("Content A")
        hash2 = compute_hash("Content B")

        assert hash1 != hash2

    def test_scraped_page_hash_validation(
        self,
        parser: SpringDocParser,
        sample_spring_html: str,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Test ScrapedPage validates content hash on creation."""
        scraped_page = parser.parse(
            html=sample_spring_html,
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            version=spring_boot_version,
        )

        # Try to create ScrapedPage with wrong hash - should fail
        with pytest.raises(ValidationError):
            ScrapedPage(
                url="https://docs.spring.io/test/",
                module=SpringModule.BOOT,
                version=spring_boot_version,
                title="Test",
                raw_html="<html></html>",
                markdown_content="# Test",
                content_hash="0000000000000000000000000000000000000000000000000000000000000000",
            )
