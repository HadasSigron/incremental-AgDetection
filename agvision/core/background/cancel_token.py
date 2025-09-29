# core/background/cancel_token.py
import threading

class CancelToken:
    """
    Cancelable flag/event abstraction.
    A wrapper around threading.Event — safe to use in a multi-thread environment.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        """
        Marks that the run was canceled.
        Can be called externally (e.g., from the UI or another Worker).
        """
        self._event.set()

    def is_set(self) -> bool:
        """
        Returns True if cancellation has been triggered.
        Should be checked inside loops/calculations within runners.
        """
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """
        (Optional) Waits until cancellation is triggered.
        Useful if you want to block until canceled, with an optional timeout.
        """
        return self._event.wait(timeout)
