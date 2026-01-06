# src/rag/services/base.py
"""Base service classes."""
from abc import ABC, abstractmethod
from typing import Any
from rag.logging_config import get_logger


class BaseService(ABC):
    """Abstract base class for services."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Check service health."""
        pass