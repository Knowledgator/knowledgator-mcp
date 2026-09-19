.PHONY: help build push check-ssh dev-install dev-sse dev-stdio clean docker-compose-up docker-compose-down

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make build               Build production Docker image with SSH forwarding (Podman/Docker)"
	@echo "  make push                Push built Docker image to remote registry (GHCR/GitLab)"
	@echo "  make check-ssh           Verify SSH agent status"
	@echo "  make dev-install         Install local development environment via Poetry"
	@echo "  make dev-sse             Run local MCP server in SSE mode via Poetry (port 8000)"
	@echo "  make dev-stdio           Run local MCP server in Stdio mode via Poetry"
	@echo "  make docker-compose-up   Start MCP server using Docker Compose"
	@echo "  make docker-compose-down Stop Docker Compose services"
	@echo "  make clean               Clean build artifacts and caches"

# Check SSH Agent status and print hint if no keys are loaded
check-ssh:
	@if [ -z "$$SSH_AUTH_SOCK" ]; then \
		echo "❌ Error: SSH_AUTH_SOCK environment variable is not set."; \
		echo "💡 Start ssh-agent first using: eval \$$(ssh-agent -s)"; \
		exit 1; \
	fi
	@if ssh-add -l >/dev/null 2>&1; then \
		echo "🔑 Active SSH Agent identities:"; \
		ssh-add -l; \
	else \
		echo "⚠️ Warning: No active SSH keys found in SSH agent."; \
		echo "💡 Hint: Please load your GitHub SSH key into the agent before building:"; \
		echo "         ssh-add ~/.ssh/<your-key-file>"; \
	fi

# Build production container image using Dockerfile (single canonical source of truth)
build: check-ssh
	podman build --ssh default=$$SSH_AUTH_SOCK -t knowledgator-mcp:latest .

# Tag and push built container image to remote registry (GHCR / GitLab Registry)
push:
	@if [ -z "$$IMAGE_REF" ]; then \
		read -p "Enter Target Image Ref (e.g. ghcr.io/Knowledgator/knowledgator-mcp:latest): " IMAGE_REF; \
	fi; \
	if [ -z "$$REGISTRY_USER" ]; then \
		read -p "Enter Registry Username: " REGISTRY_USER; \
	fi; \
	if [ -z "$$REGISTRY_PASSWORD" ]; then \
		read -sp "Enter Registry PAT Token: " REGISTRY_PASSWORD; echo ""; \
	fi; \
	podman tag knowledgator-mcp:latest $$IMAGE_REF && \
	echo "$$REGISTRY_PASSWORD" | podman login -u $$REGISTRY_USER --password-stdin $$(echo $$IMAGE_REF | cut -d/ -f1) && \
	podman push $$IMAGE_REF

# Install dev environment with Poetry
dev-install:
	poetry install
	@echo "Poetry development environment installed successfully!"

# Run local MCP server via Poetry in SSE mode
dev-sse:
	poetry run knowledgator-mcp --transport sse --host 127.0.0.1 --port 8000

# Run local MCP server via Poetry in Stdio mode
dev-stdio:
	poetry run knowledgator-mcp --transport stdio

# Docker Compose shortcuts
docker-compose-up:
	docker-compose up -d --build

docker-compose-down:
	docker-compose down

# Clean build artifacts
clean:
	rm -rf .venv *.egg-info build dist .pytest_cache __pycache__ knowledgator_mcp/__pycache__
