# Runner Interface (BaseRunner)

Unified contract for all task-specific runners (Detector, Segmenter, Classifier).
Ensures consistent signatures, structured outputs, and easy extensibility.

## Contract

```python
from typing import Dict, Any, Protocol
from core.types.runner_result import RunnerResult

class BaseRunner(Protocol):
    def run(self, dataset, config: Dict[str, Any], cancel_token) -> RunnerResult:
        ...
```

### Return format
- metrics: Dict[str, float] — e.g., {"mAP@0.5:0.95": 0.318, "IoU": 0.71, "F1": 0.82}
- artifacts: Dict[str, str] — paths/URIs to outputs, e.g.:
  - report_json
  - metrics_csv
  - confusion_matrix_png
  - pred_masks_dir
  - pred_boxes_json
  - pred_images_dir

## Adding a New Runner
1. Implement run(dataset, config, cancel_token) -> RunnerResult.
2. Check cancel_token.is_set() in long loops/stages; on cancel, return a valid (possibly empty) result.
3. Keep artifact keys informative and consistent.
4. Save outputs under runs/{task}/{domain}/{benchmark}/{timestamp}/...
5. (Optional) Add your runner to the UI/registry.

## Usage Example

```python
from core.runners.dummy_runner import DummyRunner
from core.background.cancel_token import CancelToken

runner = DummyRunner()
tok = CancelToken()
result = runner.run(dataset=[], config={"batch_size": 8, "device": "cpu"}, cancel_token=tok)
print(result["metrics"])
print(result["artifacts"])
```

## Contract Tests (pytest)

```bash
pytest -k test_base_runner_contract
```

### Tests ensure:
- Return object has metrics and artifacts dicts.
- Metric values are numeric (float-like).
- Artifacts are strings (paths/URIs).
- Runner respects cancellation.
