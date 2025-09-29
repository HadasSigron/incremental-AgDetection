# ui/components/benchmark_picker.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
from ui.state.app_state import AppState


class BenchmarkPicker(QWidget):
    """Benchmark selection panel (ID -> load YAML -> summary)."""
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self) -> None:
        self.setObjectName("panel")
        self.setStyleSheet("#panel{border:1px solid #ccc;border-radius:8px;padding:10px}")
        layout = QVBoxLayout(self)

        title = QLabel("Benchmark")
        title.setStyleSheet("font-weight:600")
        layout.addWidget(title)

        self.ed_id = QLineEdit()
        self.ed_id.setPlaceholderText("Benchmark ID (according to current domain)")
        layout.addWidget(self.ed_id)

        btn = QPushButton("Load")
        btn.clicked.connect(self._load)
        layout.addWidget(btn)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setPlaceholderText("Benchmark summary will appear here…")
        layout.addWidget(self.txt_summary)

    def on_domain_changed(self) -> None:
        """Clear selection when domain (tab) changes."""
        self.state.benchmark_id = None
        self.state.benchmark_yaml = None
        self.state.benchmark_summary = {}
        self.ed_id.clear()
        self.txt_summary.clear()

    def _load(self) -> None:
        bench_id = self.ed_id.text().strip()
        if not bench_id:
            self.txt_summary.setPlainText("Please enter a Benchmark ID.")
            return

        # Resolve YAML path by convention, then fallback to treating ID as a full path.
        domain_folder = (self.state.domain or "P2").lower()
        yaml_path = Path(f"configs/benchmarks/{domain_folder}/{bench_id}.yaml")
        if not yaml_path.exists():
            yaml_path = Path(bench_id)  # allow full path input
        if not yaml_path.exists():
            self.txt_summary.setPlainText(f"YAML not found:\n{yaml_path}")
            return

        try:
            data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            self.txt_summary.setPlainText(f"Failed to parse YAML:\n{e}")
            return

        name = data.get("name", bench_id)
        task = data.get("task", "?")
        dataset = data.get("dataset", {}) or {}
        img_dir = dataset.get("img_dir") or dataset.get("images") or "?"
        mask_or_ann = dataset.get("mask_dir") or dataset.get("ann_file") or "?"

        self.state.benchmark_id = bench_id
        self.state.benchmark_yaml = str(yaml_path)
        # also set taskType so UI (metrics table) knows which columns to show
        task_lower = str(task).strip().lower()
        task_map = {"segmentation": "Segmentation", "detection": "Detection", "classification": "Classification"}
        self.state.taskType = task_map.get(task_lower, None)

        self.state.benchmark_summary = {
            "name": name,
            "task": task,
            "img_dir": img_dir,
            "mask_or_ann": mask_or_ann,
            "domain": self.state.domain,
            "yaml": str(yaml_path),
        }

        self.txt_summary.setPlainText(
            f"Name: {name}\nTask: {task}\nImages: {img_dir}\nMasks/Annotations: {mask_or_ann}\nYAML: {yaml_path}"
        )
