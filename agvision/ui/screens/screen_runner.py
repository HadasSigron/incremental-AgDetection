from __future__ import annotations

from typing import Optional, Dict, Any
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThreadPool
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QSplitter, QMessageBox,
    QCheckBox, QHBoxLayout, QPushButton
)

from ui.state.app_state import AppState
from ui.components.model_uploader import ModelUploader
from ui.components.benchmark_picker import BenchmarkPicker
from ui.components.run_controls import RunControls
from ui.components.progress_panel import ProgressPanel
from ui.components.metrics_table import MetricsTable
from ui.workers.eval_worker import EvalWorker, WorkerSignals
from core.background.cancel_token import CancelToken
from ui.components.image_processing_dialog import ImageProcessingDialog


# ---------------- metric name mapping ----------------

def _segmentation_aliases() -> dict[str, str]:
    return {
        "miou": "mIoU",
        "mean_iou": "mIoU",
        "pixel_accuracy": "Pixel Accuracy",
        "pa": "Pixel Accuracy",
        "dice": "Dice/F1",
        "dice_f1": "Dice/F1",
        "f1": "Dice/F1",
        "precision": "Precision",
        "recall": "Recall",
    }


def _normalize_metrics_for_table(task: str, metrics: dict, artifacts: dict) -> dict[str, Any]:
    """
    Map runner metric keys -> the exact headers used by MetricsTable.
    Keep 'duration' inside metrics for the table's 'Run time' column.
    """
    task = (task or "").lower()
    src = dict(metrics or {})
    out: dict[str, Any] = {}

    # Prefer duration from metrics; if missing, take runtime from artifacts
    duration = src.get("duration")
    if duration is None:
        rt = artifacts.get("runtime_sec")
        if rt is not None:
            duration = str(rt)

    if task == "segmentation":
        alias = _segmentation_aliases()
        for k, v in src.items():
            if k == "duration":
                continue
            dk = alias.get(str(k).lower())
            if dk is None:
                dk = k if any(c.isupper() for c in str(k)) else str(k).replace("_", " ").title()
            out[dk] = v
    else:
        out.update(src)

    if duration is not None:
        out["duration"] = duration
    return out


# ---------------- main screen ----------------

class RunnerScreen(QWidget):
    """
    Main runner screen: left inputs, right outputs.
    """

    image_process_requested = pyqtSignal(object)

    def __init__(self, state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state = state
        self.cancel = CancelToken()
        self.worker: Optional[EvalWorker] = None
        self.signals: Optional[WorkerSignals] = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        # Tabs P1..P4 (domain)
        self.tabs = QTabWidget()
        for name in ("P1", "P2", "P3", "P4"):
            self.tabs.addTab(QWidget(), name)
        self.tabs.currentChanged.connect(self._on_domain_change)
        layout.addWidget(self.tabs)

        # Split: left inputs, right outputs
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left_l = QVBoxLayout(left)
        right = QWidget(); right_l = QVBoxLayout(right)

        # Left controls
        self.model_uploader = ModelUploader(self.state)
        self.image_proc_btn = QPushButton("Image Processing", self)
        self.image_proc_btn.clicked.connect(self._on_open_image_processing)
        self.benchmark_picker = BenchmarkPicker(self.state)
        self.run_controls = RunControls(self.state)
        self.run_controls.run_requested.connect(self._on_run)
        self.run_controls.stop_requested.connect(self._on_stop)

        left_l.addWidget(self.model_uploader)
        left_l.addWidget(self.image_proc_btn)
        left_l.addWidget(self.benchmark_picker)
        left_l.addWidget(self.run_controls)
        left_l.addStretch(1)

        # Right: log + filter + table
        filter_bar = QWidget(); hb = QHBoxLayout(filter_bar)
        hb.setContentsMargins(0, 0, 0, 0)
        self.chk_show_current_only = QCheckBox("Show only current benchmark")
        self.chk_show_current_only.setChecked(True)
        self.chk_show_current_only.stateChanged.connect(self._on_filter_toggle)
        hb.addWidget(self.chk_show_current_only); hb.addStretch(1)

        self.progress_panel = ProgressPanel()
        self.metrics_table = MetricsTable(self.state)

        right_l.addWidget(self.progress_panel)
        right_l.addWidget(filter_bar)
        right_l.addWidget(self.metrics_table)

        split.addWidget(left); split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        # Initialize table filter
        self.metrics_table.set_current_benchmark(getattr(self.state, "benchmark_id", None))
        self.image_process_requested.connect(self._process_image_hook)

    # ---------------- domain/filter ----------------

    @pyqtSlot(int)
    def _on_domain_change(self, idx: int) -> None:
        self.state.domain = ["P1", "P2", "P3", "P4"][idx]
        self.benchmark_picker.on_domain_changed()
        self.metrics_table.set_current_benchmark(None)

    def _on_filter_toggle(self, _state: int) -> None:
        mode = "current" if self.chk_show_current_only.isChecked() else "all"
        self.metrics_table.set_filter_mode(mode)
        self.metrics_table.set_current_benchmark(getattr(self.state, "benchmark_id", None))

    # ---------------- image processing ----------------

    def _on_open_image_processing(self) -> None:
        dlg = ImageProcessingDialog(parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            algo = dlg.selected_algorithm()
            self.image_process_requested.emit(algo)

    def _process_image_hook(self, algorithm: object) -> None:
        self.state.image_processing_algorithm = algorithm if algorithm is not None else None
        self.state.image_processing_params = {}
        try:
            self.progress_panel.on_log(
                f"Image processing: {self.state.image_processing_algorithm or 'None'}"
            )
        except Exception:
            pass

    # ---------------- run lifecycle ----------------

    def _require_ready(self) -> bool:
        if not self.state.model_file:
            QMessageBox.warning(self, "Error", "Please upload a model first.")
            return False
        if not self.state.benchmark_yaml:
            QMessageBox.warning(self, "Error", "Please enter and load a Benchmark ID (YAML).")
            return False
        return True

    def _on_run(self) -> None:
        # Pick a default table layout until the worker returns the real task
        self.metrics_table.refresh_metric_keys_from_state()
        self.metrics_table.set_task_type(self.state.taskType or "Detection")

        if not self._require_ready():
            return
        if self.worker is not None:
            QMessageBox.information(self, "Run in progress", "A run is already active.")
            return

        self.progress_panel.reset()

        yaml_path = self.state.benchmark_yaml

        # Auto-model from AppState
        fmt = (self.state.model_format or "").strip().lower()
        if fmt in (".onnx",):
            fmt = "onnx"
        auto_model = {
            "format": fmt or "onnx",
            "path": self.state.model_file,
            "imgsz": 640,
        }

        # Domain override from tabs
        dom_map = {"P1": "weeds", "P2": "plant_disease", "P3": "fruit", "P4": "uav"}
        dom = dom_map.get(self.state.domain, "plant_disease")

        # Device/batch overrides + domain
        dev = (self.state.device or "Auto").strip().lower()
        dev = "auto" if dev == "auto" else ("cpu" if dev == "cpu" else "gpu")
        overrides = {
            "domain": dom,
            "eval": {
                "device": dev,
                "batch_size": int(self.state.batch_size or 2),
            }
        }

        self.cancel = CancelToken()
        self.worker = EvalWorker(
            benchmark_yaml=str(yaml_path),
            auto_model=auto_model,
            overrides=overrides,
            cancel_token=self.cancel,
        )
        self.signals = self.worker.signals
        self.signals.progress.connect(self.progress_panel.on_progress)
        self.signals.message.connect(self.progress_panel.on_log)
        self.signals.error.connect(self.progress_panel.on_log)

        # IMPORTANT: add row on 'result' (not on 'finished'), so there is one row per run
        self.signals.result.connect(self._on_worker_result)
        self.signals.finished.connect(self._on_finished)

        QThreadPool.globalInstance().start(self.worker)

        self.run_controls.set_running(True)
        self.metrics_table.set_current_benchmark(getattr(self.state, "benchmark_id", None))
        self.metrics_table.set_filter_mode(
            "current" if self.chk_show_current_only.isChecked() else "all"
        )




    def _on_stop(self) -> None:
        if self.worker is not None:
            self.cancel.set()
            self.progress_panel.on_log("\u23F9 Stop requested…")

    # ---------------- worker callbacks ----------------

    def _on_worker_result(self, payload: dict) -> None:
        """
        Called when EvalWorker emits the result payload:
          - switch table to correct task
          - normalize metric keys to table headers
          - append exactly one row
        """
        try:
            task = str(payload.get("task", self.state.taskType or "Segmentation"))
            raw_metrics = dict(payload.get("metrics", {}) or {})
            artifacts = dict(payload.get("artifacts", {}) or {})

            # Choose table layout by task (affects header columns)
            self.metrics_table.set_task_type(task)

            # Map metric keys and keep 'duration'
            metrics_for_table = _normalize_metrics_for_table(task, raw_metrics, artifacts)

            # Meta columns
            meta = {
                "model": self.state.model_file,
                "format": self.state.model_format or "?",
                "arch": self.state.model_arch or "?",
            }

            # NOTE: append_result takes positional args (meta, metrics, benchmark_id)
            self.metrics_table.append_result(
                meta,
                metrics_for_table,
                getattr(self.state, "benchmark_id", None),
            )
        except Exception as e:
            self.progress_panel.on_log(f"[UI] Result handling error: {e}")

    def _on_finished(self, _result: Dict[str, Any]) -> None:
        """
        End-of-run cleanup. We do NOT append rows here (already handled in _on_worker_result).
        """
        self.run_controls.set_running(False)
        self.worker = None
        self.signals = None
        self.progress_panel.on_log("\u2705 Completed.")

    # ---------------- external API ----------------

    def on_task_changed(self) -> None:
        """Call this after TaskSelectScreen sets state.taskType."""
        try:
            self.metrics_table.set_task_type(self.state.taskType or "Detection")
        except Exception:
            pass
