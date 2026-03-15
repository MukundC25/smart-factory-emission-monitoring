"""FastAPI shared dependencies."""

from __future__ import annotations

from backend.utils.data_loader import DataLoader

_loader: DataLoader | None = None


def get_data_loader() -> DataLoader:
    """Return the singleton DataLoader for dependency injection.

    Returns:
        DataLoader: Global singleton instance.
    """
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader
