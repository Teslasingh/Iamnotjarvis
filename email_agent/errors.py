"""Domain exceptions — mapped to HTTP responses only in the API layer."""

from __future__ import annotations


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class AuthorizationError(AppError):
    def __init__(self, message: str = "Gmail is not authorized. Click Connect Gmail first.") -> None:
        super().__init__(message, status_code=401)


class ValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class ExternalServiceError(AppError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message, status_code=status_code)


class ConflictError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)
