"""
Main application window for MediaDubFlow.

Provides the outer desktop shell:
- Top application bar with logo/brand and settings trigger.
- QStackedWidget hosting screens (Home, Project Wizard, Processing Workspace).
- Dedicated AsyncBridge QThread for non-blocking asynchronous operations.
- Bottom status bar reflecting worker concurrency and hardware acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config import settings
from mediadubflow.core.job_manager import JobManager
from mediadubflow.database.repository import get_episode
from mediadubflow.database.session import get_session
from mediadubflow.gui.async_bridge import AsyncBridge
from mediadubflow.gui.dialogs import EpisodeDetailPanel, SettingsDialog
from mediadubflow.gui.screens.home_screen import HomeScreen
from mediadubflow.gui.screens.project_wizard_screen import ProjectWizardScreen
from mediadubflow.gui.screens.workspace_screen import WorkspaceScreen
from mediadubflow.gui.styles import GLOBAL_QSS
from mediadubflow.models.orm import Episode
from mediadubflow.utils.ffmpeg import kill_all_ffmpeg_processes

if TYPE_CHECKING:
    from mediadubflow.config.settings import SettingsValidationReport


class MediaDubFlowApp(QMainWindow):
    """
    Top-level application window.

    Navigation between screens is handled by swapping the central
    stacked widget's current index.
    """

    def __init__(
        self,
        validation_report: SettingsValidationReport | None = None,
    ) -> None:
        super().__init__()
        self.validation_report = validation_report
        self.setWindowTitle("MediaDubFlow — Khmer Localization Studio")
        self.setMinimumSize(960, 580)
        self.resize(1120, 720)
        self.setStyleSheet(GLOBAL_QSS)

        # Initialize background async worker thread
        self._bridge = AsyncBridge(self)
        self._bridge.start()
        self._bridge.wait_until_ready()

        # Initialize JobManager and start async dispatch loop in background
        self._job_manager = JobManager(self)
        self._bridge.run_async(self._job_manager.start())

        self._active_project_id: int | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        root_widget = QWidget()
        root_widget.setObjectName("root")
        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Header Bar
        header = self._build_header_bar()
        root_layout.addWidget(header)

        # 2. Main Stacked Widget
        self._stack = QStackedWidget()

        # Index 0: Home Screen
        self._home_screen = HomeScreen(bridge=self._bridge)
        self._home_screen.new_project_requested.connect(self._on_new_project_clicked)
        self._home_screen.project_selected.connect(self._on_project_selected)
        self._stack.addWidget(self._home_screen)

        # Index 1: New Project Wizard Screen
        self._wizard_screen = ProjectWizardScreen(bridge=self._bridge)
        self._wizard_screen.cancelled.connect(self._on_wizard_cancelled)
        self._wizard_screen.project_created.connect(self._on_project_created)
        self._stack.addWidget(self._wizard_screen)

        # Index 2: Processing Workspace Screen
        self._workspace_screen = WorkspaceScreen(
            bridge=self._bridge,
            job_manager=self._job_manager,
        )
        self._workspace_screen.back_requested.connect(self._on_workspace_back_clicked)
        self._workspace_screen.episode_selected.connect(self._on_episode_selected)
        self._stack.addWidget(self._workspace_screen)

        root_layout.addWidget(self._stack, stretch=1)

        self.setCentralWidget(root_widget)

        # 3. Status Bar
        self._setup_status_bar()

    def _build_header_bar(self) -> QWidget:
        header = QWidget()
        header.setObjectName("headerBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)

        # Brand / Title
        brand_label = QLabel("MediaDubFlow")
        brand_label.setObjectName("brandLabel")
        layout.addWidget(brand_label)

        layout.addSpacing(16)

        # Active Project Breadcrumb
        self._breadcrumb_label = QLabel("Home")
        self._breadcrumb_label.setObjectName("breadcrumbLabel")
        layout.addWidget(self._breadcrumb_label)

        layout.addStretch(1)

        # Settings Button
        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_settings.clicked.connect(self._on_settings_clicked)
        layout.addWidget(self._btn_settings)

        return header

    def _setup_status_bar(self) -> None:
        status_bar = self.statusBar()
        if status_bar is None:
            return

        device_info = f"Hardware: {settings.device.upper()}"
        concurrency_info = f"Workers: {settings.max_concurrent_episodes}"
        db_info = "Database: Connected"

        status_bar.showMessage(f"{device_info}  |  {concurrency_info}  |  {db_info}")

    def _on_new_project_clicked(self) -> None:
        """Handle request to create a new project."""
        logger.info("Opening New Project Wizard")
        self._wizard_screen.reset()
        self._breadcrumb_label.setText("New Project Wizard")
        self._stack.setCurrentIndex(1)

    def _on_wizard_cancelled(self) -> None:
        """Handle user cancellation in project wizard."""
        logger.info("Project Wizard cancelled, returning to Home")
        self._breadcrumb_label.setText("Home")
        self._stack.setCurrentIndex(0)
        self._home_screen.refresh_projects()

    def _on_project_created(self, project_id: int) -> None:
        """Handle successful project creation from wizard."""
        logger.info("Project id={} created from Wizard, loading workspace", project_id)
        self._active_project_id = project_id
        self._breadcrumb_label.setText(f"Project #{project_id}")
        self._workspace_screen.load_project(project_id)
        self._stack.setCurrentIndex(2)

    def _on_project_selected(self, project_id: int) -> None:
        """Handle selection of an existing project from the HomeScreen."""
        logger.info("Project id={} selected from HomeScreen, loading workspace", project_id)
        self._active_project_id = project_id
        self._breadcrumb_label.setText(f"Project #{project_id}")
        self._workspace_screen.load_project(project_id)
        self._stack.setCurrentIndex(2)

    def _on_workspace_back_clicked(self) -> None:
        """Handle back button from Workspace to return to Home."""
        logger.info("Returning to Home screen from Workspace")
        self._breadcrumb_label.setText("Home")
        self._stack.setCurrentIndex(0)
        self._home_screen.refresh_projects()

    def _on_episode_selected(self, episode_id: int) -> None:
        """Handle episode inspection request from Workspace."""
        logger.info("Episode id={} selected for review", episode_id)

        async def _fetch_episode() -> Episode | None:
            async with get_session() as session:
                return await get_episode(session, episode_id)

        def _show_panel(ep: Episode | None) -> None:
            if ep:
                dlg = EpisodeDetailPanel(ep, self)
                dlg.exec()

        self._bridge.run_async(_fetch_episode(), on_success=_show_panel)

    def _on_settings_clicked(self) -> None:
        """Open the Settings dialog and update status bar upon saving."""
        logger.info("Opening Settings dialog")
        dlg = SettingsDialog(self)
        if dlg.exec():
            self._setup_status_bar()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Ensure background jobs and thread shutdown cleanly."""
        kill_all_ffmpeg_processes()
        self._job_manager.stop(cancel_active=True)
        self._bridge.stop()
        super().closeEvent(event)
