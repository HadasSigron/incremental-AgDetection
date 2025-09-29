# agvision/core/runners/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from core.config.schema import AppConfig  # Strongly-typed config
from core.types.runner_result import RunnerResult

@runtime_checkable
class BaseRunner(Protocol):
    """
    Unified contract for all runners (Classifier/Detector/Segmenter/…).

    Each runner MUST:
      - Accept a normalized dataset adapter (see core/datasets/*).
      - Accept a strongly-typed AppConfig (see core/config/schema.py).
      - Respect a cooperative cancel_token (object exposing is_set()).
      - Return a RunnerResult with ONLY numeric metrics and artifact paths.

    NOTE:
      Keep runner logic UI/server-agnostic. The UI builds AppConfig,
      and runners rely solely on cfg.* values (batch_size, device, etc.).
    """

    def run(self, dataset: Any, cfg: AppConfig, cancel_token: Any | None) -> RunnerResult:
        """
        Execute evaluation of a model on the given dataset.

        Args:
            dataset: Normalized dataset adapter (classification/detection/segmentation).
            cfg:     Strongly-typed configuration object (task/domain/dataset/model/eval/paths).
            cancel_token: Cooperative cancel flag. If .is_set() -> True, runner should stop ASAP.

        Returns:
            RunnerResult: {"metrics": Dict[str, float], "artifacts": Dict[str, str]}
        """
        ...
# Backwards-friendly alias (some code may import Runner instead of BaseRunner)
Runner = BaseRunner
