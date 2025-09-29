# -*- coding: utf-8 -*-
# Ensure RunnerScreen has tabs and switches domain correctly,
# and requires model + benchmark before running.

import sys
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QMessageBox

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ui.state.app_state import AppState
from ui.screens.screen_runner import RunnerScreen

def test_runner_tabs_and_domain(qtbot, monkeypatch):
    state = AppState(taskType="Detection")
    w = RunnerScreen(state)
    qtbot.addWidget(w)

    # Initial domain should be P1
    assert state.domain == "P1"

    # Switch to P3
    w.tabs.setCurrentIndex(2)
    assert state.domain == "P3"

def test_run_requires_model_and_benchmark(qtbot, monkeypatch):
    state = AppState(taskType="Detection")
    w = RunnerScreen(state)
    qtbot.addWidget(w)

    # Stub QMessageBox to avoid blocking dialogs during tests
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    # No model/benchmark -> should block
    w._on_run()
    assert w.worker is None

    # With model but no benchmark -> still block
    state.model_file = __file__
    w._on_run()
    assert w.worker is None
