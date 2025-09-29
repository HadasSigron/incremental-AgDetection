from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

# --- Pydantic v1/v2 compatibility ---------------------------------------------
try:
    # Pydantic v2
    from pydantic import BaseModel, Field, field_validator, model_validator
    V2 = True
except Exception:
    # Pydantic v1 fallback
    from pydantic import BaseModel, Field, validator, root_validator  # type: ignore
    V2 = False


# ----------------------------- Enums -------------------------------------------

class TaskType(str, Enum):
    classification = "classification"
    detection = "detection"
    segmentation = "segmentation"


class Domain(str, Enum):
    """High-level application domains (tabs in the UI)."""
    weeds = "weeds"                  # P1
    plant_disease = "plant_disease"  # P2
    fruit = "fruit"                  # P3
    uav = "uav"                      # P4


class DatasetKind(str, Enum):
    """Normalized dataset adapter kinds under core/datasets/."""
    classification_folder = "classification_folder"  # ImageFolder-like
    coco_detection = "coco_detection"                # COCO bbox
    voc_detection = "voc_detection"                  # Pascal VOC
    yolo_detection = "yolo_detection"                # YOLO txt/yaml
    masks_segmentation = "masks_segmentation"        # PNG/RLE masks


class ModelFormat(str, Enum):
    """Model loading backends supported by core/models/."""
    timm = "timm"             # PyTorch timm classification
    torchscript = "torchscript"
    onnx = "onnx"
    tensorrt = "tensorrt"
    script = "script"         # user-provided Python plugin (file + entrypoint)
    pytorch = "pytorch"       # plain *.pt/*.pth via Python

# ----------------------------- Models ------------------------------------------

class PathsConfig(BaseModel):
    """
    Repository-local paths (relative to repo root or absolute).
    Used by runners/reporters to store artifacts and to resolve datasets.
    """
    data_root: Path = Field(default=Path("data"))
    runs_root: Path = Field(default=Path("runs"))
    registry_db: Path = Field(default=Path("data/registry.sqlite"))

    if not V2:
        @validator("data_root", "runs_root", "registry_db", pre=True)
        def _expand(cls, v):
            return Path(str(v)).expanduser()
    else:
        @field_validator("data_root", "runs_root", "registry_db", mode="before")
        def _expand(cls, v):
            return Path(str(v)).expanduser()


class DatasetConfig(BaseModel):
    """
    Dataset adapter selection and parameters.

    The 'kind' field determines which adapter under core/datasets/* will be used.
    Fields like 'root', 'img_dir', 'ann_file' are interpreted by the adapter.

    Examples:
    - classification_folder: root=<images dir>, split=[train/val/test] folders.
    - coco_detection: root=<dataset root>, img_dir=<images>, ann_file=<instances.json>.
    - yolo_detection: root=<dataset root>, img_dir=<images>, ann_file=<data.yaml or labels>.
    - masks_segmentation: root=<dataset root>, img_dir=<images>, mask_dir=<masks>.
    """
    kind: DatasetKind
    root: Path
    split: str = "test"
    img_dir: Optional[Path] = None
    ann_file: Optional[Path] = None
    mask_dir: Optional[Path] = None
    class_map: Optional[Path] = Field(
        default=None, description="Optional class-name mapping file"
    )
    input_size: int = 224  # Used by classification; runners may ignore otherwise
    extra: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    if not V2:
        @validator("root", "img_dir", "ann_file", "mask_dir", "class_map", pre=True)
        def _expand_paths(cls, v):
            return None if v is None else Path(str(v)).expanduser()
    else:
        @field_validator("root", "img_dir", "ann_file", "mask_dir", "class_map", mode="before")
        def _expand_paths(cls, v):
            return None if v is None else Path(str(v)).expanduser()


class ModelConfig(BaseModel):
    """
    Model loading parameters. The 'format' selects an implementation from core/models/*.

    For 'timm' classification, put the architecture in 'name', e.g. 'resnet50'.
    For exported models (torchscript/onnx/tensorrt), put the artifact path in 'weights'.
    For 'pytorch' checkpoints (*.pt/*.pth loaded via Python), keep 'weights' and optional 'arch/name'.
    For 'script' plugins, use 'extra' to provide script_path/entrypoint/kwargs.
    """
    format: ModelFormat = ModelFormat.timm
    name: Optional[str] = Field(default=None, description="Model architecture or alias")
    weights: Optional[Path] = Field(default=None, description="Path to weights/artifact")
    num_classes: Optional[int] = None
    device: str = "auto"
    imgsz: Optional[int] = None
    extra: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)

    if not V2:
        # ---- Pydantic v1 ----
        @validator("weights", pre=True)
        def _expand_weights(cls, v):
            return None if v is None else Path(str(v)).expanduser()

        @validator("format", pre=True)
        def _normalize_format_v1(cls, v):
            mapping = {
                "torch/pytorch": "pytorch",
                "pytorch":       "pytorch",
                "torch":         "pytorch",
                "pt":            "pytorch",
                "ts":            "torchscript",
            }
            s = str(v).strip().lower()
            return mapping.get(s, s)
    else:
        # ---- Pydantic v2 ----
        @field_validator("weights", mode="before")
        def _expand_weights(cls, v):
            return None if v is None else Path(str(v)).expanduser()

        @field_validator("format", mode="before")
        def _normalize_format(cls, v):
            mapping = {
                "torch/pytorch": "pytorch",
                "pytorch":       "pytorch",
                "torch":         "pytorch",
                "pt":            "pytorch",
                "ts":            "torchscript",
            }
            s = str(v).strip().lower()
            return mapping.get(s, s)


class EvalConfig(BaseModel):
    """
    Evaluation/runtime parameters that affect how a runner executes.
    """
    batch_size: int = 16
    num_workers: int = 4
    device: str = "auto"           # "cpu" | "cuda" | "mps" | "auto"
    mixed_precision: bool = False  # Enable autocast where applicable
    seed: int = 42
    metrics: List[str] = Field(
        default_factory=list,
        description="Metric names requested by the UI; defaults filled per-task"
    )
    # Optional task-specific knobs (thresholds, NMS, etc.)
    params: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """
    Logging/telemetry options (kept minimal; extend as needed).
    """
    level: str = "INFO"
    progress_interval_s: float = 0.5
    save_predictions: bool = True  # If true, runners should emit predictions.jsonl


class AppConfig(BaseModel):
    """
    Full application configuration object consumed by the UI worker and runners.
    """
    task: TaskType
    domain: Domain
    benchmark_name: str
    dataset: DatasetConfig
    model: ModelConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # --- Sanity rules binding dataset kinds to tasks ---------------------------

    if not V2:
        @root_validator
        def _check_task_dataset_compat(cls, values):
            task: TaskType = values.get("task")
            ds: DatasetConfig = values.get("dataset")
            if task and ds:
                if task == TaskType.classification and ds.kind != DatasetKind.classification_folder:
                    raise ValueError("Classification requires 'classification_folder' dataset kind.")
                if task == TaskType.detection and ds.kind not in {
                    DatasetKind.coco_detection, DatasetKind.voc_detection, DatasetKind.yolo_detection
                }:
                    raise ValueError("Detection requires COCO/VOC/YOLO detection dataset kind.")
                if task == TaskType.segmentation and ds.kind != DatasetKind.masks_segmentation:
                    raise ValueError("Segmentation requires 'masks_segmentation' dataset kind.")
            return values
    else:
        @model_validator(mode="after")
        def _check_task_dataset_compat(self):
            if self.task == TaskType.classification and self.dataset.kind != DatasetKind.classification_folder:
                raise ValueError("Classification requires 'classification_folder' dataset kind.")
            if self.task == TaskType.detection and self.dataset.kind not in {
                DatasetKind.coco_detection, DatasetKind.voc_detection, DatasetKind.yolo_detection
            }:
                raise ValueError("Detection requires COCO/VOC/YOLO detection dataset kind.")
            if self.task == TaskType.segmentation and self.dataset.kind != DatasetKind.masks_segmentation:
                raise ValueError("Segmentation requires 'masks_segmentation' dataset kind.")
            return self
