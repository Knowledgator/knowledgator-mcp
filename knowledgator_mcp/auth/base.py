from abc import ABC, abstractmethod
from typing import Dict


class BaseAuthProvider(ABC):
    """Abstract Base Class for Authentication Providers."""

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return the dictionary of authorization headers."""
        pass

    @abstractmethod
    def get_base_url(self) -> str:
        """Return the API base URL."""
        pass
