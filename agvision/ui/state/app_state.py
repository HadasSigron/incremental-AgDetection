from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any

TaskType = Literal["Detection", "Classification", "Segmentation"]
DomainType = Literal["P1", "P2", "P3", "P4"]


@dataclass
class AppState:
    """
    Single source of truth for cross-screen and cross-component UI state.
    This state object is passed into screens/components and mutated by them.
    """

    # High-level selection
    taskType: Optional[TaskType] = None
    domain: DomainType = "P1"

    # Model info (populated by ModelUploader)
    model_file: Optional[str] = None
    model_size: Optional[int] = None
    model_mtime: Optional[float] = None

    # Auto-detection toggles/fields (autodetect later via core.registry.autodetect)
    model_auto: bool = True
    model_format: Optional[str] = None   # e.g., "onnx" | "torchscript" | "script"
    model_arch: Optional[str] = None
    labels_file: Optional[str] = None

    # Benchmark selection (populated by BenchmarkPicker)
    benchmark_id: Optional[str] = None
    benchmark_yaml: Optional[str] = None  # <---- NEW: resolved YAML path
    benchmark_summary: Dict[str, Any] = field(default_factory=dict)

    # Run configuration (populated by RunControls)
    device: str = "Auto"   # "Auto" | "CPU" | "GPU"
    batch_size: int = 8
    metric_preset: str = "default"

    # Image processing selection (set by the Image Processing dialog)
    # None means "no processing" (current phase).
    image_processing_algorithm: Optional[str] = None
    image_processing_params: Dict[str, Any] = field(default_factory=dict)
