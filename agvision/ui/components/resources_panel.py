# """Resources panel (CPU/RAM/GPU) (placeholder)."""

from __future__ import annotations
try:
    import psutil  # Optional: if missing, panel will show N/A
except Exception:
    psutil = None
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class ResourcesPanel(QWidget):
    """Simple CPU/RAM monitor ticking every 500ms."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._start_timer()

    def _build(self):
        self.setObjectName("panel")
        self.setStyleSheet("#panel{border:1px solid #ccc;border-radius:8px;padding:10px}")
        layout = QVBoxLayout(self)
        title = QLabel("משאבים (CPU/RAM)")
        title.setStyleSheet("font-weight:600")
        layout.addWidget(title)
        self.lbl = QLabel("CPU: N/A | RAM: N/A")
        layout.addWidget(self.lbl)

    def _start_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    def _on_tick(self):
        if psutil is None:
            self.lbl.setText("CPU: N/A | RAM: N/A (install psutil)")
            return
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self.lbl.setText(f"CPU: {cpu:.0f}% | RAM: {mem:.0f}%")
        except Exception:
            self.lbl.setText("CPU: N/A | RAM: N/A")
