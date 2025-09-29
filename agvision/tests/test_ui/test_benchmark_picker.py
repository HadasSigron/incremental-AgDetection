# -*- coding: utf-8 -*-
# Verify loading a benchmark ID updates AppState and clears on domain change.

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ui.state.app_state import AppState
from ui.components.benchmark_picker import BenchmarkPicker

def test_benchmark_picker_load_and_domain_reset(qtbot):
    state = AppState(domain="P1")
    w = BenchmarkPicker(state)
    qtbot.addWidget(w)

    w.ed_id.setText("bench-123")
    w._load()
    assert state.benchmark_id == "bench-123"
    assert "name" in state.benchmark_summary

    # Simulate domain tab change
    w.on_domain_changed()
    assert state.benchmark_id is None
    assert state.benchmark_summary == {}
