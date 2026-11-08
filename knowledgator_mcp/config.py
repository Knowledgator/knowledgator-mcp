import json
import os
from typing import Optional, Union
from pydantic import field_validator
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
    platform_frontend_url: str = "https://platform.knowledgator.com"

    # Transport Security & DNS Rebinding Protection
    enable_dns_rebinding_protection: bool = True
    allowed_hosts: Union[list[str], str] = [
        "localhost",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:*",
        "[::1]",
        "[::1]:*",
    ]
    allowed_origins: Union[list[str], str] = []

    def get_effective_api_key(self) -> Optional[str]:
        """Return API key from settings or fall back to environment variables."""
        return self.api_key or os.getenv("KG_MCP_API_KEY") or os.getenv("KG_API_KEY")

    @field_validator("allowed_hosts", "allowed_origins", mode="before")
    @classmethod
    def parse_string_list(cls, v: object) -> list[str]:
        """Allow comma-separated strings or JSON-serialized lists in environment variables."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if str(item).strip()]
        return []


settings = ServerConfig()
