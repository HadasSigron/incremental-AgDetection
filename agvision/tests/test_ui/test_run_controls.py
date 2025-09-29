# -*- coding: utf-8 -*-
# Verify signals and state updates from RunControls.

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ui.state.app_state import AppState
from ui.components.run_controls import RunControls

def test_run_controls_emits_and_sets_state(qtbot):
    state = AppState()
    w = RunControls(state)
    qtbot.addWidget(w)

    # Change device and batch
    w.cmb_device.setCurrentText("CPU")
    w.spn_batch.setValue(16)

    with qtbot.waitSignal(w.run_requested, timeout=1000):
        w._emit_run()

    assert state.device == "CPU"
    assert state.batch_size == 16

    with qtbot.waitSignal(w.stop_requested, timeout=1000):
        w._emit_stop()
