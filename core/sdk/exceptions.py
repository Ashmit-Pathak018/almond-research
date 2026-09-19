"""
Project Almond V3 — SDK Exceptions
Typed error hierarchy for external API interactions.
"""

from __future__ import annotations
from typing import Optional, Any


class AlmondError(Exception):
    """Base exception for all Almond SDK errors."""
    pass


class AlmondConnectionError(AlmondError):
    """Raised when connection to Almond service fails."""
    pass


class AlmondAPIError(AlmondError):
    """Raised when Almond service returns an HTTP error."""

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: Optional[str] = None,
        detail: Optional[Any] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type or "api_error"
        self.detail = detail

    def __repr__(self) -> str:
        return f"<AlmondAPIError status={self.status_code} type='{self.error_type}' detail={self.detail}>"


class AlmondNotFoundError(AlmondAPIError):
    """Raised on HTTP 404 (resource not found)."""
    pass


class AlmondValidationError(AlmondAPIError):
    """Raised on HTTP 400 or 422 (validation error)."""
    pass


class AlmondConflictError(AlmondAPIError):
    """Raised on HTTP 409 (state conflict)."""
    pass
