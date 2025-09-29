# """Screen 1: task selection (placeholder)."""


from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QRadioButton, QPushButton, QButtonGroup
)

from ui.state.app_state import AppState


class TaskSelectScreen(QWidget):
    """Screen #1: choose the task type. Emits continue_requested to navigate onward."""
    continue_requested = pyqtSignal()

    def __init__(self, state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        title = QLabel("Choose a task")
        title.setObjectName("h1")
        title.setStyleSheet("#h1{font-size:28px;font-weight:600;margin:16px 0}")
        layout.addWidget(title)

        self.group = QButtonGroup(self)
        row = QHBoxLayout()
        for name in ("Detection", "Classification", "Segmentation"):
            rb = QRadioButton(name)
            rb.setProperty("task", name)  # store value on the widget
            self.group.addButton(rb)
            row.addWidget(rb)
        layout.addLayout(row)

        btn = QPushButton("Continue")
        btn.clicked.connect(self._on_continue)
        layout.addWidget(btn)
        layout.addStretch(1)

    def _on_continue(self):
        # Default to Detection if nothing is selected
        checked = self.group.checkedButton()
        self.state.taskType = checked.property("task") if checked else "Detection"
        self.continue_requested.emit()
