"""
Home Screen — Initial welcoming screen for MediaDubFlow.

Provides:
- Hero branding and action to create a new project.
- Real-time search filter for recent projects.
- Full CRUD operations:
  - Create: "+ New Project" launches Project Wizard.
  - Read: Lists recent projects with episode counts, progress indicators, and dates.
  - Update: "Edit Project" via three-dot menu opens EditProjectDialog.
  - Delete: "Delete Project" via three-dot menu prompts confirmation and removes project.
- Action column with sleek three-dot button (···) triggering an accessible context menu.
- Generous row height accommodating progress bar, status badges, and action buttons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.database.repository import list_projects
from mediadubflow.database.session import get_session
from mediadubflow.gui.dialogs.edit_project_dialog import EditProjectDialog
from mediadubflow.models.orm import EpisodeStatus
from mediadubflow.services.project_service import (
    delete_project_with_files,
    update_project_details,
)

if TYPE_CHECKING:
    from mediadubflow.gui.async_bridge import AsyncBridge
    from mediadubflow.models.orm import Project


class HomeScreen(QWidget):
    """
    Landing screen presenting '+ New Project', Search bar, and the Recent Projects table.

    Emits:
        new_project_requested: Emitted when '+ New Project' is clicked.
        project_selected: Emitted with project_id when an existing project is opened.
    """

    new_project_requested = Signal()
    project_selected = Signal(int)

    def __init__(self, bridge: AsyncBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._all_projects: list[Project] = []
        self._filtered_projects: list[Project] = []

        self._setup_ui()
        self.refresh_projects()

    @property
    def _projects(self) -> list[Project]:
        """Backwards compatibility accessor for project lists."""
        return self._filtered_projects

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(48, 40, 48, 40)
        main_layout.setSpacing(20)

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
        main_layout.addSpacing(12)

        # 2. Section Header & Search Bar
        section_widget = QWidget()
        section_layout = QHBoxLayout(section_widget)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(12)

        self._lbl_section_title = QLabel("Recent Projects")
        self._lbl_section_title.setObjectName("sectionTitle")
        section_layout.addWidget(self._lbl_section_title)

        section_layout.addStretch(1)

        self._search_input = QLineEdit()
        self._search_input.setObjectName("projectSearchInput")
        self._search_input.setPlaceholderText("Search recent projects...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.setFixedWidth(260)
        self._search_input.textChanged.connect(self._on_search_text_changed)
        section_layout.addWidget(self._search_input)

        main_layout.addWidget(section_widget)

        # 3. Empty State Label
        self._empty_label = QLabel(
            "No recent projects found. Click '+ New Project' to get started."
        )
        self._empty_label.setStyleSheet("color: #64748b; font-size: 13px; padding: 32px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setVisible(False)
        main_layout.addWidget(self._empty_label)

        # 4. Recent Projects Table (5 Columns with three-dot action menu)
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Project Name",
            "Episodes",
            "Status",
            "Created",
            "Action",
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 170)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 140)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(4, 75)

        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(54)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.verticalScrollBar().setSingleStep(16)
        self._table.cellDoubleClicked.connect(self._on_table_cell_double_clicked)

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
        self._all_projects = projects
        self._apply_search_filter()

    def _on_search_text_changed(self, _text: str) -> None:
        """Filter the projects table in real time as operator types."""
        self._apply_search_filter()

    def _apply_search_filter(self) -> None:
        """Apply query string filter and refresh table display."""
        query = self._search_input.text().strip().lower()

        if not query:
            self._filtered_projects = list(self._all_projects)
        else:
            self._filtered_projects = [
                p
                for p in self._all_projects
                if query in p.name.lower() or query in p.source_folder.lower()
            ]

        total = len(self._all_projects)
        filtered_count = len(self._filtered_projects)

        # Update section header counter
        if query and total > 0:
            self._lbl_section_title.setText(f"Recent Projects ({filtered_count} of {total})")
        elif total > 0:
            self._lbl_section_title.setText(f"Recent Projects ({total})")
        else:
            self._lbl_section_title.setText("Recent Projects")

        # Empty state management
        if total == 0:
            self._table.setVisible(False)
            self._empty_label.setText(
                "No recent projects found. Click '+ New Project' to get started."
            )
            self._empty_label.setVisible(True)
            return

        if filtered_count == 0:
            self._table.setVisible(False)
            self._empty_label.setText(
                f"No projects matching '{self._search_input.text().strip()}'. Try a different search term."
            )
            self._empty_label.setVisible(True)
            return

        self._empty_label.setVisible(False)
        self._table.setVisible(True)
        self._populate_table(self._filtered_projects)

    def _populate_table(self, projects: list[Project]) -> None:
        """Render project rows with details, status pill, progress, and three-dot action button."""
        self._table.setRowCount(len(projects))

        for row, project in enumerate(projects):
            self._table.setRowHeight(row, 54)

            # Col 0: Project Name
            name_item = QTableWidgetItem(project.name)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            name_item.setToolTip(
                f"Source: {project.source_folder}\nOutput: {project.output_folder}"
            )
            self._table.setItem(row, 0, name_item)

            # Col 1: Episodes Count
            ep_count = len(project.episodes) if project.episodes else 0
            ep_item = QTableWidgetItem(f"{ep_count} Episodes" if ep_count != 1 else "1 Episode")
            ep_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table.setItem(row, 1, ep_item)

            # Col 2: Status & Progress Cell
            status_cell = self._create_status_cell(project)
            self._table.setCellWidget(row, 2, status_cell)

            # Col 3: Created Date
            created_str = (
                project.created_at.strftime("%Y-%m-%d %H:%M")
                if getattr(project, "created_at", None)
                else "—"
            )
            created_item = QTableWidgetItem(created_str)
            created_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table.setItem(row, 3, created_item)

            # Col 4: Action (Three-Dot Button)
            action_cell = self._create_action_cell(project)
            self._table.setCellWidget(row, 4, action_cell)

    def _create_status_cell(self, project: Project) -> QWidget:
        """Create a clean cell displaying the status pill and mini progress bar if active."""
        container = QWidget()
        status_text, variant, pct = self._calculate_project_status(project)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        badge = self._create_status_badge(status_text, variant)
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignCenter)

        if variant == "processing":
            prog_bar = QProgressBar()
            prog_bar.setProperty("class", "tableProgressBar")
            prog_bar.setRange(0, 100)
            prog_bar.setValue(pct)
            prog_bar.setTextVisible(False)
            prog_bar.setFixedHeight(6)
            prog_bar.setFixedWidth(130)
            layout.addWidget(prog_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        return container

    def _create_action_cell(self, project: Project) -> QWidget:
        """Create a compact cell containing the three-dot button (···)."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_more = QPushButton("···")
        btn_more.setProperty("class", "threeDotButton")
        btn_more.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_more.setToolTip("Actions")
        btn_more.setFixedSize(36, 30)
        btn_more.clicked.connect(lambda _, b=btn_more, p=project: self._show_project_menu(b, p))

        layout.addWidget(btn_more)
        return container

    def _show_project_menu(self, btn: QPushButton, project: Project) -> None:
        """Display dropdown menu anchored to the three-dot button."""
        menu = QMenu(self)

        action_open = menu.addAction("Open Project")
        action_open.triggered.connect(lambda: self.project_selected.emit(project.id))

        action_edit = menu.addAction("Edit Project")
        action_edit.triggered.connect(lambda: self._on_edit_project_clicked(project))

        menu.addSeparator()

        action_delete = menu.addAction("Delete Project")
        action_delete.triggered.connect(lambda: self._on_delete_project_clicked(project))

        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _on_table_cell_double_clicked(self, row: int, _col: int) -> None:
        """Double clicking a project row opens the project workspace."""
        if 0 <= row < len(self._filtered_projects):
            self.project_selected.emit(self._filtered_projects[row].id)

    def _on_edit_project_clicked(self, project: Project) -> None:
        """Open EditProjectDialog and save updated fields asynchronously."""
        dlg = EditProjectDialog(project, parent=self)
        if dlg.exec():
            updated = dlg.get_updated_values()
            p_id = project.id

            async def _update() -> Project:
                async with get_session() as session:
                    res = await update_project_details(
                        session,
                        p_id,
                        name=updated["name"],
                        output_folder=updated.get("output_folder"),
                        target_language=updated.get("target_language"),
                    )
                    await session.commit()
                    return res

            self._bridge.run_async(
                _update(),
                on_success=lambda _: self.refresh_projects(),
                on_error=lambda exc: QMessageBox.warning(
                    self,
                    "Update Project Failed",
                    f"Could not update project: {exc}",
                ),
            )

    def _on_delete_project_clicked(self, project: Project) -> None:
        """Prompt confirmation and delete project and cache directories asynchronously."""
        confirm = QMessageBox.question(
            self,
            "Delete Project",
            f"Are you sure you want to delete '{project.name}'?\n\n"
            "This will remove the project, episode records, and cached files from MediaDubFlow.\n"
            "Source videos will not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        p_id = project.id

        async def _delete() -> bool:
            async with get_session() as session:
                res = await delete_project_with_files(session, p_id)
                await session.commit()
                return res

        self._bridge.run_async(
            _delete(),
            on_success=lambda _: self.refresh_projects(),
            on_error=lambda exc: QMessageBox.warning(
                self,
                "Delete Project Failed",
                f"Could not delete project: {exc}",
            ),
        )

    @staticmethod
    def _create_status_badge(text: str, variant: str) -> QLabel:
        """Construct a styled status badge label with crisp borders."""
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if variant == "completed":
            lbl.setStyleSheet(
                "border: 1px solid #16a34a; color: #16a34a; font-weight: 600; font-size: 11px; padding: 2px 8px; border-radius: 4px;"
            )
        elif variant == "processing":
            lbl.setStyleSheet(
                "border: 1px solid #2563eb; color: #2563eb; font-weight: 600; font-size: 11px; padding: 2px 8px; border-radius: 4px;"
            )
        elif variant == "queued":
            lbl.setStyleSheet(
                "border: 1px solid #cbd5e1; color: #64748b; font-weight: 500; font-size: 11px; padding: 2px 8px; border-radius: 4px;"
            )
        else:
            lbl.setStyleSheet(
                "border: 1px solid #e2e8f0; color: #94a3b8; font-weight: 500; font-size: 11px; padding: 2px 8px; border-radius: 4px;"
            )
        return lbl

    @staticmethod
    def _calculate_project_status(project: Project) -> tuple[str, str, int]:
        """Derive friendly status label, badge type, and progress percentage."""
        if not project.episodes:
            return "Empty", "empty", 0

        total = len(project.episodes)
        done_count = sum(1 for ep in project.episodes if ep.status == EpisodeStatus.DONE)
        pct = int((done_count / total) * 100)

        if done_count == total:
            return "Completed", "completed", 100

        active_count = sum(
            1
            for ep in project.episodes
            if ep.status not in (EpisodeStatus.PENDING, EpisodeStatus.DONE, EpisodeStatus.FAILED)
        )
        if active_count > 0:
            return f"Processing ({pct}%)", "processing", pct

        if done_count > 0:
            return f"{done_count} of {total} Done", "processing", pct

        return "Queued", "queued", 0
