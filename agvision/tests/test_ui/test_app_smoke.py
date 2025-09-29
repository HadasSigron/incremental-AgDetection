# -*- coding: utf-8 -*-
# Basic smoke test that the app creates and shows two screens.

import sys
from pathlib import Path
import importlib

from PyQt6.QtWidgets import QApplication

def test_app_importable():
    """Module import should not crash and main symbols should exist."""
    # Ensure repo root is in sys.path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    app_mod = importlib.import_module("ui.app")
    assert hasattr(app_mod, "App")
    assert hasattr(app_mod, "main")
