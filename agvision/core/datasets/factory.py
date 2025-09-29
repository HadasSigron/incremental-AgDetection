# core/datasets/factory.py
from __future__ import annotations

from typing import Callable, Dict, Any

DatasetBuilder = Callable[[Any], Any]

# Global registry of dataset builders
_DATASET_REGISTRY: Dict[str, DatasetBuilder] = {}


def register_dataset(kind: str, builder: DatasetBuilder) -> None:
    """
    Register a dataset builder under a canonical, lowercase key.
    """
    _DATASET_REGISTRY[kind.strip().lower()] = builder


def normalize_kind(kind_obj: Any) -> str:
    """
    Accept Enum (with .value) or string and normalize to a canonical lowercase string.
    Supports values like 'DatasetKind.classification_folder' or 'classification_folder'.
    """
    if hasattr(kind_obj, "value"):
        return str(kind_obj.value).lower()
    s = str(kind_obj).lower()
    return s.split(".", 1)[1] if s.startswith("datasetkind.") else s


def build_dataset_adapter(cfg) -> Any:
    """
    Build an adapter by kind via the registry.
    Raises a helpful error listing registered kinds if not found.
    """
    kind = normalize_kind(getattr(cfg.dataset, "kind", ""))
    if not kind:
        raise ValueError("dataset.kind is empty — cannot build dataset adapter.")
    try:
        builder = _DATASET_REGISTRY[kind]
    except KeyError:
        registered = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset kind: {kind}. Registered kinds: [{registered}]"
        )
    return builder(cfg)


# Import the package __init__ to trigger self-registration of adapters.
# (It is safe to import at module end; it won't recurse infinitely.)
import core.datasets as _datasets  # noqa: F401
