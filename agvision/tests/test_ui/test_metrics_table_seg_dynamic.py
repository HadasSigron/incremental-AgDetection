# tests/test_ui/test_metrics_table_seg_dynamic.py
"""
Qt smoke for MetricsTable dynamic columns in Segmentation.
We simulate a worker payload with requested metrics from YAML and verify that
the table shows these columns (headers) and a row contains the values.

Requires pytest-qt.
"""

from __future__ import annotations
import pytest
from types import SimpleNamespace

pytest.importorskip("PyQt6")  # skip if PyQt6 not available in CI

from PyQt6.QtWidgets import QApplication
from ui.components.metrics_table import MetricsTable


class _LooseState:
    """
    Minimal fake 'state' to satisfy MetricsTable __init__(state).
    Any missing attribute access returns None, so the widget can be instantiated
    without the real application state.
    """
    def __getattr__(self, name: str):
        return None


def _norm(s: str) -> str:
    """
    Normalize table header or metric key to compare robustly:
    - lowercase
    - remove non-alphanumeric
    Example:
      "Pixel Accuracy" -> "pixelaccuracy"
      "mIoU"           -> "miou"
      "dice"           -> "dice"
    """
    s = s.lower()
    return "".join(ch for ch in s if ch.isalnum())


def test_metrics_table__seg_dynamic_columns(qtbot):
    # Create widget with a dummy state
    state = _LooseState()
    w = MetricsTable(state)
    qtbot.addWidget(w)

    # Simulated worker payload (Segmentation)
    payload = {
        "task": "Segmentation",
        "benchmark": "tiny-seg",
        "requested_metrics": ["miou", "pixel_accuracy", "dice"],  # from YAML
        "metrics": {"miou": 0.5, "pixel_accuracy": 0.8, "dice": 0.6, "duration": 0.01},
        "artifacts": {"run_dir": "runs/x", "runtime_sec": "0.010"},
        "meta": {"model": "fake", "format": "script", "arch": "fake", "duration": 0.01},
    }

    # Append result as the UI would do
    # If your method name differs (e.g., add_row/append_row), adjust here accordingly.
    w.append_result(meta=payload["meta"], metrics=payload["metrics"], requested=payload["requested_metrics"])

    # Headers shown in the table
    headers = [w.table.horizontalHeaderItem(i).text() for i in range(w.table.columnCount())]
    headers_norm = { _norm(h) for h in headers }

    # Expected (normalized) keys for requested metrics
    expected = { _norm(k) for k in payload["requested_metrics"] }  # {"miou","pixelaccuracy","dice"}

    # Ensure all requested metrics appear as columns (after friendly-label mapping)
    assert expected.issubset(headers_norm), f"Missing columns for: {sorted(expected - headers_norm)}"

    # One row with our values
    assert w.table.rowCount() == 1
