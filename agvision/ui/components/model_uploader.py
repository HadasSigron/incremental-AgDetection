from __future__ import annotations
from typing import Optional
from pathlib import Path
from PyQt6.QtCore import Qt
import re

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QCheckBox, QFormLayout, QLineEdit
)

from ui.state.app_state import AppState


class ModelUploader(QWidget):
    """Model selection panel with an Auto toggle for format/arch detection."""
    def __init__(self, state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        self.setObjectName("panel")
        self.setStyleSheet("#panel{border:1px solid #ccc;border-radius:8px;padding:10px}")
        layout = QVBoxLayout(self)

        title = QLabel("Model")
        title.setStyleSheet("font-weight:600")
        layout.addWidget(title)

        btn = QPushButton("Upload model…")
        btn.clicked.connect(self._pick)
        layout.addWidget(btn)

        # Auto mode: lock inputs and fill from autodetection
        self.chk_auto = QCheckBox("Auto")
        self.chk_auto.setChecked(True)
        self.chk_auto.stateChanged.connect(self._on_auto_changed)
        layout.addWidget(self.chk_auto)

        form = QFormLayout()
        self.ed_format = QLineEdit()
        self.ed_format.setPlaceholderText("Format, e.g., ONNX/YOLO/timm…")
        self.ed_arch = QLineEdit()
        self.ed_arch.setPlaceholderText("Architecture, e.g., resnet50/unet/yolov8n…")
        self.ed_format.setEnabled(False)
        self.ed_arch.setEnabled(False)
        form.addRow("Format:", self.ed_format)
        form.addRow("Arch:", self.ed_arch)
        layout.addLayout(form)

        self.lbl_info = QLabel("")
        self.lbl_info.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.lbl_info)

    def _on_auto_changed(self, _):
        auto = self.chk_auto.isChecked()
        self.state.model_auto = auto
        # In auto mode, lock fields; in manual mode, unlock
        self.ed_format.setEnabled(not auto)
        self.ed_arch.setEnabled(not auto)

    def _pick(self):
        # Open file dialog and persist selection to AppState
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model file",
            "",
            "Models (*.onnx *.pt *.pth *.engine);;All files (*)"
        )
        if not path:
            return
        p = Path(path)
        self.state.model_file = str(p)
        self.state.model_display = p.name            

        self.state.model_size = p.stat().st_size if p.exists() else None
        self.state.model_mtime = p.stat().st_mtime if p.exists() else None

        if self.state.model_auto:
            fmt = self._canonical_format(self._guess_format(p.suffix.lower()))
            arch = self._guess_arch_from_name(p.stem)

            self.state.model_format = fmt
            self.state.model_arch = arch
            self.ed_format.setText("Torch/PyTorch" if fmt == "pytorch" else (fmt or ""))
            self.ed_arch.setText(arch or "")
        else:
            fmt_manual = self._canonical_format(self.ed_format.text())
            self.state.model_format = fmt_manual
            self.state.model_arch = (self.ed_arch.text() or None)

            # אופציונלי: להציג בחזרה תווית יפה
            if fmt_manual == "pytorch":
                self.ed_format.setText("Torch/PyTorch")
                self.lbl_info.setText(f"{p.name} • {self.state.model_size or '?'} bytes")

    @staticmethod
    def _guess_format(suffix: str) -> Optional[str]:
        """Infer canonical format from file extension."""
        mapping = {
            ".pt": "pytorch",      # היה "Torch/PyTorch" → עכשיו קנוני: pytorch
            ".pth": "pytorch",
            ".onnx": "onnx",
            ".engine": "tensorrt",
            # ".xml": "openvino"   # לא נתמך בסכמה אצלך, נשאיר None
        }
        return mapping.get(suffix)



    @staticmethod
    def _canonical_format(s: Optional[str]) -> Optional[str]:
        """Map user/alias values to canonical schema values."""
        if not s:
            return None
        m = {
            "torch/pytorch": "pytorch",  # את בחרת לתמוך בפורמט pytorch רשמי
            "pytorch": "pytorch",
            "torch": "pytorch",
            "pt": "pytorch",
            "script": "script",
            "ts": "torchscript",
            "torchscript": "torchscript",
            "onnx": "onnx",
            "tensorrt": "tensorrt",
            "timm": "timm",
        }
        return m.get(str(s).strip().lower())

    @staticmethod
    def _canonical_format(s: Optional[str]) -> Optional[str]:
        """Map user/alias values to canonical schema values."""
        if not s:
            return None
        m = {
            "torch/pytorch": "pytorch",
            "pytorch": "pytorch",
            "torch": "pytorch",
            "pt": "pytorch",
            "script": "script",
            "ts": "torchscript",
            "torchscript": "torchscript",
            "onnx": "onnx",
            "tensorrt": "tensorrt",
            "timm": "timm",
        }
        return m.get(str(s).strip().lower())

    @staticmethod
    def _guess_format(suffix: str) -> Optional[str]:
        """Infer canonical format from file extension."""
        mapping = {
            ".pt": "pytorch",
            ".pth": "pytorch",
            ".onnx": "onnx",
            ".engine": "tensorrt",
            # ".xml": "openvino"  # לא נתמך בסכמה
        }
        return mapping.get(suffix)

    @staticmethod
    def _guess_arch_from_name(name: str) -> Optional[str]:
        """
        Heuristically infer architecture from filename (best-effort).
        """
        n = name.lower()

        # YOLO (v5/v8...) + אופציונלי "-seg"
        m = re.search(r"(yolov[0-9]+[nxslmxy]?)", n)
        if m:
            arch = m.group(1)
            if "seg" in n and not arch.endswith("-seg"):
                arch += "-seg"
            return arch

        # backbones/architectures נפוצים
        patterns = [
            r"(resnet\d+)",
            r"(efficientnet[-_]?b\d+)",
            r"(unet)",
            r"(vit[-_]?b?\d*)",
            r"(swin[-_]?t|swin[-_]?b|swin[-_]?l)",
            r"(mobilenetv2|mobilenetv3)",
            r"(deeplabv3\+?)",
            r"(fasterrcnn)",
            r"(maskrcnn)",
        ]
        for pat in patterns:
            m = re.search(pat, n)
            if m:
                return m.group(1)

        # Fallback
        tokens = re.findall(r"[a-z][a-z0-9+\-]*", n)
        blacklist = {"model", "best", "final", "latest", "weights", "checkpoint", "ckpt"}
        for t in tokens:
            if t not in blacklist and not t.isdigit():
                return t

        return None

# ---------- Metrics table component ----------