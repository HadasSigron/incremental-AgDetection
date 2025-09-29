# core/runners/registry.py
from __future__ import annotations

from typing import Dict, Type, Union
from core.config.schema import TaskType  # enum: classification|detection|segmentation
from core.runners.base import BaseRunner
from .classifier_runner import ClassifierRunner
from .detector_runner import DetectorRunner
from .segmenter_runner import SegmenterRunner  # make sure this file exists and is wired

# Map logical Task -> concrete Runner class
_REGISTRY: Dict[TaskType, Type[BaseRunner]] = {
    TaskType.classification: ClassifierRunner,
    TaskType.detection:     DetectorRunner,
    TaskType.segmentation:  SegmenterRunner,
}

def get_runner(task: Union[TaskType, str]) -> BaseRunner:
    """
    Return a runner instance by task.
    Accepts TaskType enum or case-insensitive string ('segmentation'/'detection'/...).
    """
    if isinstance(task, str):
        try:
            task_enum = TaskType(task.strip().lower())
        except Exception as e:
            raise KeyError(f"Unknown task: {task!r}") from e
    else:
        task_enum = task

    try:
        cls = _REGISTRY[task_enum]
    except KeyError as e:
        raise KeyError(f"No runner registered for task={task_enum.value!r}") from e
    return cls()
