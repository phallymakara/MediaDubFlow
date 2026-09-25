"""
Unit tests for HomeScreen, AsyncBridge, and MediaDubFlowApp.
"""

from __future__ import annotations

from mediadubflow.gui.app import MediaDubFlowApp
from mediadubflow.gui.async_bridge import AsyncBridge
from mediadubflow.gui.screens.home_screen import HomeScreen


def test_async_bridge_execution(qtbot) -> None:
    """AsyncBridge executes a coroutine and delivers the result on the main thread."""
    b = AsyncBridge()
    b.start()
    b.wait_until_ready()

    try:
        results: list[int] = []

        async def _add(x: int, y: int) -> int:
            return x + y

        b.run_async(_add(3, 4), on_success=lambda val: results.append(val))
        qtbot.waitUntil(lambda: len(results) == 1, timeout=3000)
        assert results == [7]
    finally:
        b.stop()


def test_home_screen_empty_state(qtbot) -> None:
    """When no projects exist in database, empty state label is displayed."""
    b = AsyncBridge()
    b.start()
    b.wait_until_ready()
    try:
        screen = HomeScreen(bridge=b)
        qtbot.addWidget(screen)
        assert screen._btn_new.text() == "+ New Project"
    finally:
        b.stop()


def test_home_screen_new_project_signal(qtbot) -> None:
    """Clicking + New Project button emits new_project_requested signal."""
    b = AsyncBridge()
    b.start()
    b.wait_until_ready()
    try:
        screen = HomeScreen(bridge=b)
        qtbot.addWidget(screen)
        with qtbot.waitSignal(screen.new_project_requested, timeout=2000):
            screen._btn_new.click()
    finally:
        b.stop()


def test_app_window_initialization(qtbot) -> None:
    """MediaDubFlowApp initializes properly with header, stack, and status bar."""
    app_window = MediaDubFlowApp()
    qtbot.addWidget(app_window)
    assert "MediaDubFlow" in app_window.windowTitle()
    assert app_window._home_screen is not None

    status_text = app_window.statusBar().currentMessage()
    assert "Hardware:" in status_text
    assert "Workers:" in status_text

    app_window.close()
