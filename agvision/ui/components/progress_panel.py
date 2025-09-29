# """Progress panel (placeholder)."""

from __future__ import annotations
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QTextEdit


class ProgressPanel(QWidget):
    """Displays progress percentage and live logs."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        self.setObjectName("panel")
        self.setStyleSheet("#panel{border:1px solid #ccc;border-radius:8px;padding:10px}")
        layout = QVBoxLayout(self)

        title = QLabel("Progress")
        title.setStyleSheet("font-weight:600")
        layout.addWidget(title)

        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.logs = QTextEdit(); self.logs.setReadOnly(True)
        layout.addWidget(self.logs)

    def reset(self):
        self.bar.setValue(0)
        self.logs.clear()

    def on_progress(self, value: int):
        self.bar.setValue(value)

    def on_log(self, text: str):
        self.logs.append(text)
