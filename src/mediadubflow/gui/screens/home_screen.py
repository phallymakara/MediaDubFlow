"""
Home Screen — Initial welcoming screen for MediaDubFlow.

Provides:
- Hero branding and action to create a new project.
- List of recent localization projects with episode counts, statuses, and open/resume actions.
- Clean empty state when no projects exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.database.repository import list_projects
from mediadubflow.database.session import get_session
from mediadubflow.models.orm import EpisodeStatus

if TYPE_CHECKING:
    from mediadubflow.gui.async_bridge import AsyncBridge
    from mediadubflow.models.orm import Project


class HomeScreen(QWidget):
    """
    Landing screen presenting '+ New Project' and the Recent Projects table.

    Emits:
        new_project_requested: Emitted when '+ New Project' is clicked.
        project_selected: Emitted with project_id when an existing project is opened.
    """

    new_project_requested = Signal()
    project_selected = Signal(int)

    def __init__(self, bridge: AsyncBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._projects: list[Project] = []
        self._setup_ui()
        self.refresh_projects()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(48, 40, 48, 40)
        main_layout.setSpacing(24)

        # 1. Hero Section
        hero_widget = QWidget()
        hero_layout = QVBoxLayout(hero_widget)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(8)
        hero_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Localize Your Media")
        title.setObjectName("heroTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Translate, subtitle and dub your drama and movie episodes into Khmer")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._btn_new = QPushButton("+ New Project")
        self._btn_new.setObjectName("primaryButton")
        self._btn_new.setMinimumSize(200, 42)
        self._btn_new.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_new.clicked.connect(self.new_project_requested.emit)

        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addSpacing(12)
        hero_layout.addWidget(self._btn_new, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(hero_widget)
        main_layout.addSpacing(16)

        # 2. Section Header: Recent Projects
        section_header = QLabel("Recent Projects")
        section_header.setObjectName("sectionTitle")
        main_layout.addWidget(section_header)

        # 3. Empty State Label
        self._empty_label = QLabel(
            "No recent projects found. Click '+ New Project' to get started."
        )
        self._empty_label.setStyleSheet("color: #64748b; font-size: 13px; padding: 24px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setVisible(False)
        main_layout.addWidget(self._empty_label)

        # 4. Recent Projects Table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Project Name", "Episodes", "Status", "Action"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 110)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.verticalScrollBar().setSingleStep(16)

        main_layout.addWidget(self._table, stretch=1)

    def refresh_projects(self) -> None:
        """Fetch all projects from SQLite in the background and update the table."""

        async def _fetch() -> list[Project]:
            async with get_session() as session:
                return await list_projects(session)

        self._bridge.run_async(
            _fetch(),
            on_success=self._on_projects_loaded,
            on_error=lambda exc: logger.error("Failed to load recent projects: {}", exc),
        )

    def _on_projects_loaded(self, projects: list[Project]) -> None:
        """Callback on main thread when projects are loaded."""
        self._projects = projects
        self._table.setRowCount(len(projects))

        if not projects:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._empty_label.setVisible(False)
        self._table.setVisible(True)

        for row, project in enumerate(projects):
            # Name
            name_item = QTableWidgetItem(project.name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table.setItem(row, 0, name_item)

            # Episodes count
            ep_count = len(project.episodes) if project.episodes else 0
            ep_item = QTableWidgetItem(f"{ep_count} Episodes" if ep_count != 1 else "1 Episode")
            ep_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table.setItem(row, 1, ep_item)

            # Status calculation
            status_text = self._calculate_project_status(project)
            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table.setItem(row, 2, status_item)

            # Action button
            action_btn = QPushButton("Open")
            action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            action_btn.clicked.connect(lambda _, p_id=project.id: self.project_selected.emit(p_id))

            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.setContentsMargins(4, 2, 4, 2)
            cell_layout.addWidget(action_btn)
            self._table.setCellWidget(row, 3, cell_widget)

    @staticmethod
    def _calculate_project_status(project: Project) -> str:
        """Derive a friendly status summary from the project's episodes."""
        if not project.episodes:
            return "Empty"

        total = len(project.episodes)
        done_count = sum(1 for ep in project.episodes if ep.status == EpisodeStatus.DONE)
        if done_count == total:
            return "Completed"

        active_count = sum(
            1
            for ep in project.episodes
            if ep.status not in (EpisodeStatus.PENDING, EpisodeStatus.DONE, EpisodeStatus.FAILED)
        )
        if active_count > 0:
            pct = int((done_count / total) * 100)
            return f"Processing ({pct}%)"

        return f"{done_count} of {total} Done"
