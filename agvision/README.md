# AgVision

Unified application for Detection / Segmentation / Classification with a shared core.

- Runner interface + task-specific runners
- Dataset adapters (COCO/VOC/YOLO/Classification)
- Two-screen desktop UI (Task selection; Domains & Runner)
- Optional server mode (FastAPI)
- Incremental evaluation and reporting

> All code docstrings are in English by project convention.

## Runners — Usage

### Classification Runner — Quick Start

```python
from core.runners.registry import get_runner
from core.datasets.classification_folder import ClassificationFolder

# Dataset adapter (ImageFolder-like):
ds = ClassificationFolder(root_dir="D:/data/P2/flowers")

# Config — choose one of three modes:
cfg = {
  "domain": "P2",
  "benchmark_id": "flowers_v1",
  # 1) Simulation (quick check):
  "simulate": True,
  # 2) Or a real model (TorchScript):
  # "model_path": "D:/weights/resnet50_scripted.pt",
  "batch_size": 16,
  "imgsz": 224,
  "device": "cpu",
}

class Cancel:
    def is_set(self): return False

runner = get_runner("Classification")
metrics, artifacts = runner.run(ds, cfg, Cancel())
print(metrics)
print(artifacts)
