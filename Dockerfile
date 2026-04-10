# syntax=docker/dockerfile:1.7

##
# Builder stage:
# - Uses trusted slim Python base.
# - Resolves dependencies from uv.lock in frozen mode.
# - Installs runtime dependencies into /app/.venv only.
##
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HUB_OFFLINE=1
WORKDIR /app

# Install uv only in builder (not copied to runner).
RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock README.md /app/

# Create venv and install locked, prod-only dependencies.
RUN uv venv /app/.venv \
    && uv sync --frozen --no-dev --no-install-project

# Source code is copied separately so dependency layers remain cacheable.
COPY src /app/src

##
# Runner stage:
# - Minimal runtime image.
# - No uv/build toolchain copied from builder.
# - Runs as non-root user for defense in depth.
##
FROM python:3.11-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    EVERSPRING_DATA_DIR=/home/everspring/.everspring

WORKDIR /app

RUN groupadd --gid 1000 everspring \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/everspring --shell /usr/sbin/nologin everspring \
    && mkdir -p /home/everspring/.everspring

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

RUN if [ -x "/app/.venv/bin/playwright" ]; then \
        /app/.venv/bin/playwright install chromium && \
        /app/.venv/bin/playwright install-deps chromium && \
        rm -rf /root/.cache/ms-playwright/firefox-* && \
        rm -rf /root/.cache/ms-playwright/webkit-* && \
        rm -rf /var/lib/apt/lists/*; \
    fi

RUN chown -R 1000:1000 /app /home/everspring

USER 1000:1000

# Direct module execution keeps MCP stdio protocol safe (no wrapper script output).
CMD ["python", "-m", "everspring_mcp.main", "serve"]
