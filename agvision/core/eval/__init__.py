"""Evaluation package init – export main helpers."""
from .hashing import file_sig, dict_sig, model_sig, sample_key
from .store import CacheStore
from .incremental import evaluate_incremental
from .reporter import save_predictions, save_summary
from .runs import make_run_dir

__all__ = [
    "file_sig", "dict_sig", "model_sig", "sample_key",
    "CacheStore", "evaluate_incremental",
    "save_predictions", "save_summary", "make_run_dir",
]
