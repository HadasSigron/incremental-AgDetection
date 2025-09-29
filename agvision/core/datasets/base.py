# core/datasets/base.py
"""
Base dataset adapter interface for the evaluation app.

Runners expect an adapter to expose:
- class_names() -> list[str]
- num_classes() -> int
- split_exists(name: str) -> bool
- iter_samples(split: str = "val") -> Iterable[dict]
- image_count(split: str = "val") -> int
- describe() -> dict[str, any]  # paths/info for logging

Each sample dict should minimally contain:
{
    "id": int,                 # stable per-sample integer id
    "path": str,               # absolute path to image
    "label": int | None,       # class index (None if unknown/test)
}
"""
from __future__ import annotations

from typing import Dict, Any, Iterable, Protocol


class BaseAdapter(Protocol):
    @classmethod
    def from_config(cls, cfg) -> "BaseAdapter":
        """Build adapter from a typed AppConfig instance."""
        ...

    # Metadata
    def class_names(self) -> list[str]: ...
    def num_classes(self) -> int: ...

    # Splits
    def split_exists(self, name: str) -> bool: ...
    def iter_samples(self, split: str = "val") -> Iterable[Dict[str, Any]]: ...
    def image_count(self, split: str = "val") -> int: ...

    # Diagnostics
    def describe(self) -> Dict[str, Any]: ...
