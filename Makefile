.PHONY: help build push check-ssh dev-install dev-sse dev-stdio clean docker-compose-up docker-compose-down

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make build               Build production Docker image (version: $(if $(VERSION),$(VERSION),latest)) with SSH forwarding"
	@echo "  make push                Push Docker image (tags: latest $(if $(VERSION),and $(VERSION),)) to remote registry"
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

# Build production container image using Dockerfile
build: check-ssh
	@echo "🔨 Building knowledgator-mcp..."
	@if [ -n "$(VERSION)" ]; then \
		echo "🏷️ Applying tags: latest and $(VERSION)"; \
		podman build --ssh default=$$SSH_AUTH_SOCK -t knowledgator-mcp:latest -t knowledgator-mcp:$(VERSION) .; \
	else \
		echo "🏷️ Applying tag: latest"; \
		podman build --ssh default=$$SSH_AUTH_SOCK -t knowledgator-mcp:latest .; \
	fi

# Tag and push built container image to remote registry
push:
	@IMAGE_REF="$${IMAGE_REF:-$$CI_REGISTRY_IMAGE}"; \
	REGISTRY_USER="$${REGISTRY_USER:-$$CI_REGISTRY_USER}"; \
	REGISTRY_PASSWORD="$${REGISTRY_PASSWORD:-$$CI_REGISTRY_PASSWORD}"; \
	if [ -z "$$IMAGE_REF" ]; then \
		read -p "Enter Registry Image Base (e.g. ghcr.io/username/knowledgator-mcp or registry.gitlab.com/org/knowledgator-mcp): " IMAGE_REF; \
	fi; \
	BASE_REF=$$(echo "$$IMAGE_REF" | sed 's/:.*//'); \
	REGISTRY_DOMAIN=$$(echo "$$BASE_REF" | cut -d/ -f1); \
	if [ -z "$$REGISTRY_USER" ]; then \
		read -p "Enter Registry Username: " REGISTRY_USER; \
	fi; \
	if [ -z "$$REGISTRY_PASSWORD" ]; then \
		read -sp "Enter Registry PAT Token: " REGISTRY_PASSWORD; echo ""; \
	fi; \
	echo "🏷️ Tagging images for registry:"; \
	echo "   - $${BASE_REF}:latest"; \
	podman tag knowledgator-mcp:latest $${BASE_REF}:latest && \
	if [ -n "$(VERSION)" ]; then \
		echo "   - $${BASE_REF}:$(VERSION)"; \
		podman tag knowledgator-mcp:latest $${BASE_REF}:$(VERSION); \
	fi && \
	echo "🔐 Logging in to $${REGISTRY_DOMAIN}..." && \
	echo "$$REGISTRY_PASSWORD" | podman login -u $$REGISTRY_USER --password-stdin $${REGISTRY_DOMAIN} && \
	echo "🚀 Pushing image tags..." && \
	podman push $${BASE_REF}:latest && \
	if [ -n "$(VERSION)" ]; then \
		podman push $${BASE_REF}:$(VERSION); \
	fi && \
	echo "✅ Pushed image successfully!"

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
