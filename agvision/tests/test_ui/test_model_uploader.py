# -*- coding: utf-8 -*-
# Mock file dialog to simulate model selection and verify AppState updates.

import sys
from pathlib import Path
import pytest

from PyQt6.QtWidgets import QFileDialog

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ui.state.app_state import AppState
from ui.components.model_uploader import ModelUploader

def test_model_uploader_auto_mode_updates_state(qtbot, monkeypatch, tmp_path):
    state = AppState()
    w = ModelUploader(state)
    qtbot.addWidget(w)

    # Create a fake .onnx file
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"onnx")

    # Mock dialog to return our file
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(fake_model), "Models (*.onnx)"))

    # Click the "Upload model…" flow
    w._pick()

    assert state.model_file == str(fake_model)
    assert state.model_format == "ONNX"
    assert state.model_auto is True
