# """UI validators: domain/task/model/benchmark compatibility (placeholder)."""

from __future__ import annotations
from typing import Optional


def validate_model_against_task(model_format: Optional[str], task: Optional[str]) -> Optional[str]:
    """Return an error message if model_format seems incompatible with the given task; None if OK."""
    if not model_format or not task:
        return None
    if task == "Detection" and model_format not in {"ONNX", "Torch/PyTorch", "TensorRT"}:
        return "Model is not compatible with Detection (expected ONNX/PyTorch/TensorRT)."
    if task == "Segmentation" and model_format not in {"ONNX", "Torch/PyTorch"}:
        return "Model is not compatible with Segmentation (expected ONNX/PyTorch)."
    if task == "Classification" and model_format not in {"ONNX", "Torch/PyTorch", "timm"}:
        return "Model is not compatible with Classification (expected ONNX/PyTorch/timm)."
    return None


def validate_benchmark_domain(bench_domain: Optional[str], current_domain: Optional[str]) -> Optional[str]:
    """Return an error message if a benchmark belongs to a different domain; None if OK."""
    if bench_domain and current_domain and bench_domain != current_domain:
        return f"‘{bench_domain}’ does not belong to {current_domain}. Please select a matching benchmark."
    return None
