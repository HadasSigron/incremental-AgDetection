# -*- coding: utf-8 -*-
# Verify TaskSelect screen sets AppState and emits navigation signal.

import sys
from pathlib import Path
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QRadioButton

# Ensure import path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ui.state.app_state import AppState
from ui.screens.screen_task_select import TaskSelectScreen

@pytest.mark.parametrize("choice", ["Detection", "Classification", "Segmentation"])
def test_task_select_sets_state_and_emits(choice, qtbot):
    state = AppState()
    w = TaskSelectScreen(state)
    qtbot.addWidget(w)

    # Check the radio buttons and click the desired one
    radios = w.findChildren(QRadioButton)
    target = next(rb for rb in radios if rb.text() == choice)
    target.click()

    with qtbot.waitSignal(w.continue_requested, timeout=1000):
        # Click the "continue" button (the last QPushButton in layout)
        w._on_continue()

    assert state.taskType == choice
