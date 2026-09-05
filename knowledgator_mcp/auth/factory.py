import os
from knowledgator_mcp.auth.base import BaseAuthProvider
from knowledgator_mcp.auth.providers import APIKeyAuthProvider
from knowledgator_mcp.config import ServerConfig


def get_auth_provider(server_config: ServerConfig) -> BaseAuthProvider:
    """Factory function to instantiate active auth provider based on settings."""
    auth_type = (server_config.auth_type or "api_key").lower()

    if auth_type == "api_key":
        return APIKeyAuthProvider(
            api_key=server_config.get_effective_api_key(),
            base_url=server_config.base_url,
        )
    else:
        raise ValueError(f"Unsupported authentication type: {server_config.auth_type}")
