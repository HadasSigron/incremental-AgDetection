from __future__ import annotations

from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QWidget
)


class ImageProcessingDialog(QDialog):
    """
    Modal dialog for choosing an image-processing algorithm.

    Current phase:
      - Only 'None' option exists, which means "no processing".
      - The Apply button is intentionally disabled when 'None' is selected.
        (No algorithm to apply yet; UI plumbing is prepared for future additions.)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image Processing")

        layout = QFormLayout(self)

        # Combo box with a single 'None' option (no processing).
        self.combo = QComboBox(self)
        self.combo.addItem("None", userData=None)  # None == no processing
        layout.addRow("Algorithm:", self.combo)

        # Apply/Cancel buttons; Apply disabled while 'None' is selected.
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)

        # Enable Apply only if a real algorithm (not None) is selected.
        self.combo.currentIndexChanged.connect(self._on_combo_changed)

    def _on_combo_changed(self, _index: int) -> None:
        is_none = self.combo.currentData() is None
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not is_none)

    def selected_algorithm(self) -> Optional[str]:
        """Return the algorithm key, or None when 'None' is selected."""
        return self.combo.currentData()
