# Production Multi-Stage Dockerfile for Knowledgator MCP Server
# Usage with Podman/Docker:
# podman build --ssh default=$SSH_AUTH_SOCK -t knowledgator-mcp:latest .

FROM python:3.12-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install git, ssh-client, and build tools for fetching private dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Trust github.com SSH host key
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy project files
COPY pyproject.toml poetry.lock* README.md /app/
COPY knowledgator_mcp /app/knowledgator_mcp

# Install ONLY production dependencies using host SSH agent
RUN --mount=type=ssh \
    poetry install --only main

# ----------------------------------------------------
# Stage 2: Runtime Image (Minimal production runtime)
# ----------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/knowledgator_mcp /app/knowledgator_mcp
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

EXPOSE 8000

ENTRYPOINT ["python", "-m", "knowledgator_mcp.app"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
