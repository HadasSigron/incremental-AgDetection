# tests/test_ui/test_image_processing_ui.py
# -*- coding: utf-8 -*-
"""
UI tests for the Image Processing feature.

Covers:
  1) Dialog basics: 'None'-only item and disabled Apply.
  2) RunnerScreen integration: button exists and opens the dialog.
  3) Run config plumbing: image_processing.algorithm is None during Run,
     and we wait for QThread.finished (robust on Windows).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Ensure repo root on sys.path (so "ui.*" imports work)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QPushButton, QDialog

from ui.state.app_state import AppState
from ui.components.image_processing_dialog import ImageProcessingDialog
from ui.screens.screen_runner import RunnerScreen


@pytest.fixture
def app_state() -> AppState:
    """Provide a fresh AppState for each test."""
    return AppState()


def test_image_processing_dialog_none_only(qtbot):
    """
    The dialog should present a single 'None' option, and the Apply button
    must be disabled while 'None' is selected.
    """
    dlg = ImageProcessingDialog()
    qtbot.addWidget(dlg)

    dlg.show()
    qtbot.waitExposed(dlg)

    # Combo should contain exactly one item: "None" with userData=None
    assert dlg.combo.count() == 1
    assert dlg.combo.itemText(0) == "None"
    assert dlg.combo.itemData(0) is None

    # Apply should be disabled while 'None' is selected
    apply_btn = dlg.buttons.button(dlg.buttons.StandardButton.Ok)
    assert apply_btn.isEnabled() is False

    dlg.close()
    assert dlg.isVisible() is False


def test_runner_screen_has_button_and_opens_dialog(qtbot, app_state, monkeypatch):
    """
    RunnerScreen should have an 'Image Processing' button.
    Clicking it should open the dialog (verified by monkeypatching exec()).
    """
    screen = RunnerScreen(app_state)
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)

    # Find the button by text
    buttons = [b for b in screen.findChildren(QPushButton) if b.text() == "Image Processing"]
    assert buttons, "Image Processing button not found on the screen"
    btn = buttons[0]

    # Monkeypatch ImageProcessingDialog.exec to verify it was invoked
    called = {"count": 0}

    class _FakeDialog(ImageProcessingDialog):
        def exec(self):
            called["count"] += 1
            return QDialog.DialogCode.Rejected  # Cancel

    monkeypatch.setattr(
        "ui.screens.screen_runner.ImageProcessingDialog",
        _FakeDialog,
        raising=True
    )

    qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
    assert called["count"] == 1, "Dialog.exec() was not called"


def test_run_config_includes_image_processing_none(qtbot, app_state, monkeypatch):
    """
    During Run, RunnerScreen injects image_processing with algorithm=None.
    We wait for QThread.finished (not the custom signals object) to avoid
    Windows/PyQt disconnection crashes in pytest-qt.
    """
    # Speed up the worker loop
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    # Minimal ready state
    app_state.model_file = "dummy_model.onnx"
    app_state.benchmark_id = "bench-1"
    app_state.taskType = "Detection"

    screen = RunnerScreen(app_state)
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)

    # Start run
    screen._on_run()
    worker = screen.worker
    assert worker is not None, "Worker was not started"

    # Validate the injected config (hook)
    assert hasattr(worker, "config"), "Worker missing 'config' attribute"
    assert "image_processing" in worker.config
    ip = worker.config["image_processing"]
    assert isinstance(ip, dict)
    assert ip.get("algorithm") is None

    # Robust wait: QThread.finished instead of the custom WorkerSignals.finished
    with qtbot.waitSignal(worker.finished, timeout=3000, raising=True):
        pass

    # After finish, the screen cleans up its references
    assert screen.worker is None
