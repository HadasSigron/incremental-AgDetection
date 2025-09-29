# """Run/Stop controls; device/batch/metric preset (placeholder)."""


from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QComboBox, QSpinBox

from ui.state.app_state import AppState


class RunControls(QWidget):
    """Run/Stop controls + basic run configuration (device, batch size)."""
    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        title = QLabel("Run")
        title.setStyleSheet("font-weight:600")
        layout.addWidget(title)

        # Advanced controls (device, batch)
        row = QHBoxLayout()
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["Auto", "CPU", "GPU"])
        self.spn_batch = QSpinBox(); self.spn_batch.setRange(1, 1024); self.spn_batch.setValue(8)
        row.addWidget(QLabel("Device:")); row.addWidget(self.cmb_device)
        row.addWidget(QLabel("Batch:")); row.addWidget(self.spn_batch)
        layout.addLayout(row)

        # Action buttons
        row2 = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self._emit_run)
        self.btn_stop.clicked.connect(self._emit_stop)
        row2.addWidget(self.btn_run)
        row2.addWidget(self.btn_stop)
        layout.addLayout(row2)

    def _emit_run(self):
        # Persist selected config to AppState and request a run
        self.state.device = self.cmb_device.currentText()
        self.state.batch_size = int(self.spn_batch.value())
        self.run_requested.emit()

    def _emit_stop(self):
        # Request a soft stop
        self.stop_requested.emit()

    def set_running(self, running: bool):
        # Toggle buttons based on run state
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
