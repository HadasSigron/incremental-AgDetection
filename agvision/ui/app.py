# """PyQt application entry-point (placeholder)."""
from __future__ import annotations
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QStackedWidget

from ui.state.app_state import AppState
from ui.screens.screen_task_select import TaskSelectScreen
from ui.screens.screen_runner import RunnerScreen


class App(QApplication):
    def __init__(self, argv: list[str]):
        super().__init__(argv)
        self.setApplicationName("GVision")
        self.setOrganizationName("GVision-Team")

        # Global shared state for the whole UI
        self.state = AppState()

        # Screen stack: Task selection → Runner
        self.stack = QStackedWidget()
        self.stack.setWindowTitle("GVision – Model Evaluation")
        self.stack.setMinimumSize(1100, 720)

        # Screen 1 — pick task type
        self.screen_task = TaskSelectScreen(self.state)
        self.screen_task.continue_requested.connect(self._go_to_runner)
        self.stack.addWidget(self.screen_task)

        # Screen 2 — domains + run workflow
        self.screen_runner = RunnerScreen(self.state)
        self.stack.addWidget(self.screen_runner)

        self.stack.setCurrentWidget(self.screen_task)
        self.stack.show()


    def _go_to_runner(self):
        # Make sure the runner reflects the just-selected task
        if hasattr(self.screen_runner, "on_task_changed"):
            self.screen_runner.on_task_changed()
        self.stack.setCurrentWidget(self.screen_runner)

def main():
    # Allow running from repo root or from ui/
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    app = App(sys.argv)
    sys.exit(app.exec())





if __name__ == "__main__":
    main()
