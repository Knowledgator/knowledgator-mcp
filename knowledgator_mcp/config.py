import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """Configuration settings for the Knowledgator MCP server."""

    model_config = SettingsConfigDict(
        env_prefix="KG_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server_name: str = "KnowledgatorMCP"
    debug: bool = False
    auth_type: str = "api_key"
    api_key: Optional[str] = None
    base_url: str = "https://api.knowledgator.com"
    request_timeout_ms: int = 60000

    def get_effective_api_key(self) -> Optional[str]:
        """Return API key from settings or fall back to environment variables."""
        return self.api_key or os.getenv("KG_MCP_API_KEY") or os.getenv("KG_API_KEY")


settings = ServerConfig()
