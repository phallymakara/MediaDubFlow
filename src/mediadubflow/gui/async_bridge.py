"""
Async bridge running a background QThread with an asyncio event loop.

Allows the PySide6 main GUI thread to safely execute asynchronous database queries
and service functions without freezing the user interface.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from loguru import logger
from PySide6.QtCore import QObject, Qt, QThread, Signal


class AsyncBridge(QThread):
    """
    Dedicated background worker thread hosting an asyncio event loop.

    Any coroutine can be scheduled using ``run_async()`` with optional callbacks
    executed safely on the Qt main thread upon completion or failure.
    """

    result_ready = Signal(object, object)
    error_occurred = Signal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop | None = None
        self.result_ready.connect(self._handle_result_on_main, Qt.ConnectionType.QueuedConnection)
        self.error_occurred.connect(self._handle_error_on_main, Qt.ConnectionType.QueuedConnection)
        import threading

        self._started_event = threading.Event()

    def run(self) -> None:
        """Entry point for the background QThread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.debug("AsyncBridge QThread started with dedicated event loop")
        self._started_event.set()

        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            logger.debug("AsyncBridge QThread event loop closed")

    def wait_until_ready(self, timeout: float = 2.0) -> bool:
        """Wait until the background event loop is running."""
        return self._started_event.wait(timeout)

    def stop(self) -> None:
        """Stop the background asyncio event loop and wait for the thread to exit."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.wait(3000)

    def run_async(
        self,
        coro: Coroutine[Any, Any, Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """
        Schedule a coroutine in the background event loop.

        Args:
            coro: The coroutine to execute.
            on_success: Optional callback to invoke on the Qt main thread with the result.
            on_error: Optional callback to invoke on the Qt main thread with any caught Exception.
        """
        if self._loop is None or not self._loop.is_running():
            logger.error("Cannot run coroutine: AsyncBridge event loop is not running")
            if on_error:
                on_error(RuntimeError("Async worker thread is not running"))
            return

        async def _wrapper() -> None:
            try:
                result = await coro
                if on_success:
                    self.result_ready.emit(result, on_success)
            except Exception as exc:
                logger.exception("AsyncBridge background coroutine failed: {}", exc)
                if on_error:
                    self.error_occurred.emit(exc, on_error)

        asyncio.run_coroutine_threadsafe(_wrapper(), self._loop)

    @staticmethod
    def _handle_result_on_main(result: Any, callback: Callable[[Any], None]) -> None:
        """Invoked on the Qt main thread to deliver the coroutine result."""
        try:
            callback(result)
        except Exception as exc:
            logger.exception("Error in AsyncBridge on_success callback: {}", exc)

    @staticmethod
    def _handle_error_on_main(exc: Exception, error_callback: Callable[[Exception], None]) -> None:
        """Invoked on the Qt main thread to deliver the error."""
        try:
            error_callback(exc)
        except Exception as callback_exc:
            logger.exception("Error in AsyncBridge on_error callback: {}", callback_exc)
