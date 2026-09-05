from typing import Dict, Optional

from knowledgator_mcp.auth.base import BaseAuthProvider


class APIKeyAuthProvider(BaseAuthProvider):
    """Authentication provider using Knowledgator API key."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.knowledgator.com"):
        self.api_key = api_key
        self.base_url = base_url

    def get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["KnowledgatorAPI-Key"] = self.api_key
        return headers

    def get_base_url(self) -> str:
        return self.base_url

