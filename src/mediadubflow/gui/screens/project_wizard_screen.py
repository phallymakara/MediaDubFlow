"""
New Project Wizard Screen — Step-by-step media ingestion and localization setup.

Provides a 3-step workflow:
1. Media Ingestion: Select source video directory, set project title, inline validation.
2. Episode Review: Review detected episodes, toggle inclusion, inspect sizes.
3. Localization Setup: Source language selection, target language (Khmer), mode, and series glossary.

Adheres strictly to zero-shadow UI guidelines, inline error handling, and
thread-safe background execution via AsyncBridge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, override

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config.settings import PipelineOutputMode, settings
from mediadubflow.database.session import get_session
from mediadubflow.services.project_service import create_project_from_folder
from mediadubflow.utils.episode_detector import VIDEO_EXTENSIONS, detect_episodes

if TYPE_CHECKING:
    from mediadubflow.gui.async_bridge import AsyncBridge


class MediaFolderDropZone(QFrame):
    """
    Modern drag-and-drop intake zone for media directories and video files.

    Supports:
    - Drag-and-drop of directories or video files directly from file managers.
    - Click anywhere to open the system folder browser.
    - Two clean visual states: empty state with format badges and selected state
      with folder name, truncated path, and real-time video file count.
    """

    folder_dropped = Signal(Path)
    browse_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mediaDropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("hasFolder", False)
        self.setProperty("dragOver", False)

        self._setup_ui()

    def _setup_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(28, 24, 28, 24)
        self._layout.setSpacing(10)

        # 1. Empty State Container
        self._empty_container = QWidget()
        empty_layout = QVBoxLayout(self._empty_container)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(6)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel("📁")
        icon_lbl.setStyleSheet("font-size: 32px;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(icon_lbl)

        lbl_prompt = QLabel("Drag and drop your media folder here")
        lbl_prompt.setStyleSheet("font-size: 15px; font-weight: 600; color: #0f172a;")
        lbl_prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl_prompt)

        lbl_sub = QLabel("or click anywhere in this card to browse files on your computer")
        lbl_sub.setStyleSheet("font-size: 13px; color: #64748b;")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl_sub)

        empty_layout.addSpacing(6)

        btn_browse_drop = QPushButton("Browse Folder...")
        btn_browse_drop.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse_drop.setStyleSheet("padding: 6px 18px; font-size: 13px; font-weight: 500;")
        btn_browse_drop.clicked.connect(self.browse_requested.emit)
        empty_layout.addWidget(btn_browse_drop, alignment=Qt.AlignmentFlag.AlignCenter)

        empty_layout.addSpacing(8)

        formats_row = QHBoxLayout()
        formats_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        formats_row.setSpacing(6)
        formats_label = QLabel("Supported formats:")
        formats_label.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: 500;")
        formats_row.addWidget(formats_label)

        for fmt in [".mp4", ".mkv", ".mov", ".webm", ".avi"]:
            chip = QLabel(fmt)
            chip.setProperty("class", "formatTag")
            formats_row.addWidget(chip)

        empty_layout.addLayout(formats_row)
        self._layout.addWidget(self._empty_container)

        # 2. Selected State Container
        self._selected_container = QWidget()
        selected_layout = QVBoxLayout(self._selected_container)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.setSpacing(10)

        header_row = QHBoxLayout()
        status_badge = QLabel("✓ Media Folder Selected")
        status_badge.setStyleSheet(
            "background-color: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; "
            "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600;"
        )
        header_row.addWidget(status_badge)
        header_row.addStretch(1)

        btn_change = QPushButton("Change Folder")
        btn_change.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_change.setStyleSheet("padding: 4px 12px; font-size: 12px;")
        btn_change.clicked.connect(self.browse_requested.emit)
        header_row.addWidget(btn_change)
        selected_layout.addLayout(header_row)

        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        folder_icon = QLabel("📁")
        folder_icon.setStyleSheet("font-size: 28px;")
        info_row.addWidget(folder_icon)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        self._lbl_folder_name = QLabel("")
        self._lbl_folder_name.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        self._lbl_folder_path = QLabel("")
        self._lbl_folder_path.setStyleSheet("font-size: 12px; color: #64748b;")
        text_col.addWidget(self._lbl_folder_name)
        text_col.addWidget(self._lbl_folder_path)
        info_row.addLayout(text_col, stretch=1)
        selected_layout.addLayout(info_row)

        self._lbl_scan_status = QLabel("")
        self._lbl_scan_status.setStyleSheet("font-size: 12px; font-weight: 500;")
        selected_layout.addWidget(self._lbl_scan_status)

        self._layout.addWidget(self._selected_container)
        self._selected_container.setVisible(False)

    def set_folder(self, folder_path: Path | None, video_count: int | None = None) -> None:
        if folder_path is None or not folder_path.exists():
            self._empty_container.setVisible(True)
            self._selected_container.setVisible(False)
            self.setProperty("hasFolder", False)
            self.style().unpolish(self)
            self.style().polish(self)
            return

        self._empty_container.setVisible(False)
        self._selected_container.setVisible(True)
        self.setProperty("hasFolder", True)
        self.style().unpolish(self)
        self.style().polish(self)

        self._lbl_folder_name.setText(folder_path.name)
        self._lbl_folder_path.setText(str(folder_path))
        self._lbl_folder_path.setToolTip(str(folder_path))

        if video_count is not None:
            if video_count > 0:
                self._lbl_scan_status.setText(
                    f"✓ {video_count} video file(s) found ready for indexing"
                )
                self._lbl_scan_status.setStyleSheet(
                    "color: #16a34a; font-size: 12px; font-weight: 500;"
                )
            else:
                self._lbl_scan_status.setText(
                    "No supported video files found (.mp4, .mkv, .mov, .webm, .avi)"
                )
                self._lbl_scan_status.setStyleSheet(
                    "color: #dc2626; font-size: 12px; font-weight: 500;"
                )
        else:
            self._lbl_scan_status.setText("Folder ready")
            self._lbl_scan_status.setStyleSheet(
                "color: #64748b; font-size: 12px; font-weight: 500;"
            )

    @override
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    self.setProperty("dragOver", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    return
        event.ignore()

    @override
    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()

    @override
    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

        for url in event.mimeData().urls():
            if url.isLocalFile():
                p = Path(url.toLocalFile())
                target = p if p.is_dir() else p.parent
                if target.is_dir():
                    event.acceptProposedAction()
                    self.folder_dropped.emit(target)
                    return
        event.ignore()

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.pos())
            if isinstance(child, QPushButton):
                super().mousePressEvent(event)
                return
            self.browse_requested.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


class ProjectWizardScreen(QWidget):
    """
    Multi-step wizard guiding the user through media ingestion, episode filtering,
    and localization parameters.

    Emits:
        project_created: Emitted with the created project's database ID.
        cancelled: Emitted when the user chooses to cancel and return home.
    """

    project_created = Signal(int)
    cancelled = Signal()

    def __init__(self, bridge: AsyncBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge

        # Wizard State
        self._source_dir: Path | None = None
        self._detected_episodes: list[tuple[int, Path]] = []
        self._selected_episodes: list[tuple[int, Path]] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(36, 20, 36, 20)
        main_layout.setSpacing(16)

        # 1. Header & Stepper Indicator
        self._stepper_widget = self._build_stepper()
        main_layout.addWidget(self._stepper_widget)

        # 2. Stacked Pages (Step 1, Step 2, Step 3)
        self._step_stack = QStackedWidget()
        self._step1_widget = self._build_step1_media()
        self._step2_widget = self._build_step2_episodes()
        self._step3_widget = self._build_step3_localization()

        self._step_stack.addWidget(self._step1_widget)  # Index 0
        self._step_stack.addWidget(self._step2_widget)  # Index 1
        self._step_stack.addWidget(self._step3_widget)  # Index 2
        main_layout.addWidget(self._step_stack, stretch=1)

        self._update_stepper(0)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Stepper Indicator
    # -------------------------------------------------------------------------
    def _build_stepper(self) -> QWidget:
        container = QWidget()
        container.setObjectName("stepperContainer")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 16)
        layout.setSpacing(12)

        steps_data = [
            ("STEP 1", "Media Folder", "Source directory"),
            ("STEP 2", "Review Episodes", "Filter video files"),
            ("STEP 3", "Localization Setup", "Target language & mode"),
        ]

        self._step_elements: list[tuple[QLabel, QLabel, QLabel, QFrame]] = []
        self._step_connectors: list[QFrame] = []
        self._step_labels: list[QLabel] = []

        for i, (tag_text, title_text, _) in enumerate(steps_data):
            item_frame = QFrame()
            item_frame.setStyleSheet("background: transparent; border: none;")
            item_layout = QHBoxLayout(item_frame)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(10)

            # Circular badge
            badge = QLabel(str(i + 1))
            badge.setFixedSize(28, 28)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            item_layout.addWidget(badge)

            # Text labels
            text_box = QVBoxLayout()
            text_box.setSpacing(1)
            text_box.setContentsMargins(0, 0, 0, 0)

            tag_lbl = QLabel(tag_text)
            title_lbl = QLabel(title_text)

            text_box.addWidget(tag_lbl)
            text_box.addWidget(title_lbl)
            item_layout.addLayout(text_box)

            # Click handler to navigate back to completed steps
            def _make_handler(idx: int):
                def _on_click(event: QMouseEvent) -> None:
                    if idx < self._step_stack.currentIndex():
                        self._update_stepper(idx)
                    event.accept()

                return _on_click

            item_frame.mousePressEvent = _make_handler(i)

            self._step_elements.append((badge, tag_lbl, title_lbl, item_frame))
            self._step_labels.append(title_lbl)
            layout.addWidget(item_frame)

            # Connector line between steps
            if i < len(steps_data) - 1:
                connector = QFrame()
                connector.setFixedHeight(2)
                connector.setStyleSheet("background-color: #e2e8f0; border: none;")
                self._step_connectors.append(connector)
                layout.addWidget(connector, stretch=1)

        return container

    def _update_stepper(self, active_index: int) -> None:
        self._step_stack.setCurrentIndex(active_index)

        for i, (badge, tag, title, item_frame) in enumerate(self._step_elements):
            if i < active_index:
                # Completed step
                badge.setText("✓")
                badge.setStyleSheet(
                    "background-color: #f0fdf4; color: #16a34a; border: 1.5px solid #16a34a; "
                    "border-radius: 14px; font-weight: 700; font-size: 13px;"
                )
                tag.setStyleSheet("color: #16a34a; font-weight: 700; font-size: 10px;")
                title.setStyleSheet("color: #0f172a; font-weight: 600; font-size: 13px;")
                item_frame.setCursor(Qt.CursorShape.PointingHandCursor)
                item_frame.setToolTip(f"Click to return to Step {i + 1}")
            elif i == active_index:
                # Active step
                badge.setText(str(i + 1))
                badge.setStyleSheet(
                    "background-color: #2563eb; color: #ffffff; border: 1.5px solid #2563eb; "
                    "border-radius: 14px; font-weight: 700; font-size: 12px;"
                )
                tag.setStyleSheet("color: #2563eb; font-weight: 700; font-size: 10px;")
                title.setStyleSheet("color: #0f172a; font-weight: 700; font-size: 13px;")
                item_frame.setCursor(Qt.CursorShape.ArrowCursor)
                item_frame.setToolTip("")
            else:
                # Inactive / Upcoming step
                badge.setText(str(i + 1))
                badge.setStyleSheet(
                    "background-color: #f8fafc; color: #94a3b8; border: 1.5px solid #cbd5e1; "
                    "border-radius: 14px; font-weight: 600; font-size: 12px;"
                )
                tag.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 10px;")
                title.setStyleSheet("color: #94a3b8; font-weight: 500; font-size: 13px;")
                item_frame.setCursor(Qt.CursorShape.ArrowCursor)
                item_frame.setToolTip("")

        # Update connector lines
        for i, connector in enumerate(self._step_connectors):
            if i < active_index:
                connector.setStyleSheet("background-color: #16a34a; border: none;")
            else:
                connector.setStyleSheet("background-color: #e2e8f0; border: none;")

    # -------------------------------------------------------------------------
    # Step 1: Media Ingestion & Project Setup
    # -------------------------------------------------------------------------
    def _build_step1_media(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(14)

        # Scrollable form content for responsive screen heights
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.verticalScrollBar().setSingleStep(16)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(16)

        heading_box = QVBoxLayout()
        heading_box.setSpacing(4)
        title = QLabel("Select Media Folder")
        title.setObjectName("sectionTitle")
        subtitle = QLabel(
            "Choose or drop the folder containing your drama or movie video files to localize."
        )
        subtitle.setStyleSheet("color: #64748b; font-size: 13px;")
        heading_box.addWidget(title)
        heading_box.addWidget(subtitle)
        layout.addLayout(heading_box)

        # 1. Modern Interactive Drop Zone
        self._dropzone = MediaFolderDropZone()
        self._dropzone.browse_requested.connect(self._on_browse_folder)
        self._dropzone.folder_dropped.connect(self._set_folder_path)
        layout.addWidget(self._dropzone)

        # Inline folder error label directly under drop zone
        self._err_folder = QLabel("")
        self._err_folder.setStyleSheet("color: #dc2626; font-size: 12px; margin-top: 2px;")
        self._err_folder.setVisible(False)
        layout.addWidget(self._err_folder)

        # 2. Selected Folder Path Bar (Compact manual inspection / browse)
        folder_box = QVBoxLayout()
        folder_box.setSpacing(6)
        folder_lbl = QLabel("Selected Folder Path")
        folder_lbl.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        folder_box.addWidget(folder_lbl)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        self._txt_folder = QLineEdit()
        self._txt_folder.setPlaceholderText(
            "Select folder containing video files (.mp4, .mkv, .mov, .webm, .avi)"
        )
        self._txt_folder.setReadOnly(True)
        self._txt_folder.textChanged.connect(self._on_folder_text_changed)

        btn_browse = QPushButton("Browse...")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._on_browse_folder)

        folder_row.addWidget(self._txt_folder, stretch=1)
        folder_row.addWidget(btn_browse)
        folder_box.addLayout(folder_row)
        layout.addLayout(folder_box)

        # 3. Project Name Field
        name_box = QVBoxLayout()
        name_box.setSpacing(6)
        name_lbl = QLabel("Project Title")
        name_lbl.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        name_box.addWidget(name_lbl)

        self._txt_name = QLineEdit()
        self._txt_name.setPlaceholderText("Enter series or drama title")
        self._txt_name.textChanged.connect(self._clear_name_error)
        name_box.addWidget(self._txt_name)

        name_hint = QLabel(
            "Identifies this project in the workspace and names generated subtitle and audio files."
        )
        name_hint.setStyleSheet("color: #94a3b8; font-size: 12px;")
        name_box.addWidget(name_hint)

        self._err_name = QLabel("")
        self._err_name.setStyleSheet("color: #dc2626; font-size: 12px; margin-top: 2px;")
        self._err_name.setVisible(False)
        name_box.addWidget(self._err_name)
        layout.addLayout(name_box)

        layout.addStretch(1)

        scroll_area.setWidget(content_widget)
        page_layout.addWidget(scroll_area, stretch=1)

        # Actions Row (pinned at bottom)
        actions_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.cancelled.emit)

        self._btn_step1_next = QPushButton("Next: Review Episodes →")
        self._btn_step1_next.setObjectName("primaryButton")
        self._btn_step1_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_step1_next.clicked.connect(self._on_step1_next)

        actions_row.addWidget(btn_cancel)
        actions_row.addStretch(1)
        actions_row.addWidget(self._btn_step1_next)
        page_layout.addLayout(actions_row)

        return page

    def _set_folder_path(self, folder_path: Path) -> None:
        self._source_dir = folder_path
        self._txt_folder.setText(str(folder_path))
        self._clear_folder_error()

        # Auto-populate project name if empty
        if not self._txt_name.text().strip():
            self._txt_name.setText(folder_path.name.replace("_", " ").title())
            self._clear_name_error()

        self._update_folder_preview(folder_path)

    def _on_browse_folder(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Drama/Movie Folder",
            str(Path.home()),
        )
        if not selected_dir:
            return

        self._set_folder_path(Path(selected_dir))

    def _on_folder_text_changed(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            p = Path(cleaned)
            self._source_dir = p
            self._update_folder_preview(p)
        else:
            self._source_dir = None
            self._update_folder_preview(None)

    def _update_folder_preview(self, folder_path: Path | None) -> None:
        if not folder_path or not folder_path.exists():
            if hasattr(self, "_dropzone"):
                self._dropzone.set_folder(None)
            return

        try:
            count = sum(
                1
                for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
        except Exception:
            count = None

        if hasattr(self, "_dropzone"):
            self._dropzone.set_folder(folder_path, video_count=count)

    def _clear_folder_error(self) -> None:
        self._err_folder.setVisible(False)
        self._txt_folder.setStyleSheet("")

    def _clear_name_error(self) -> None:
        self._err_name.setVisible(False)
        self._txt_name.setStyleSheet("")

    def _on_step1_next(self) -> None:
        # Validate Project Name
        name = self._txt_name.text().strip()
        if not name:
            self._err_name.setText("Project title is required.")
            self._err_name.setVisible(True)
            self._txt_name.setStyleSheet("border: 1px solid #dc2626;")
            return

        # Validate Folder
        if not self._source_dir or not self._source_dir.exists():
            self._err_folder.setText("Please select a valid media folder.")
            self._err_folder.setVisible(True)
            self._txt_folder.setStyleSheet("border: 1px solid #dc2626;")
            return

        # Detect episodes
        try:
            detected = detect_episodes(self._source_dir)
        except Exception as exc:
            logger.exception("Failed to scan directory for episodes: {}", exc)
            self._err_folder.setText(f"Unable to read folder: {exc}")
            self._err_folder.setVisible(True)
            self._txt_folder.setStyleSheet("border: 1px solid #dc2626;")
            return

        if not detected:
            self._err_folder.setText(
                "No supported video files found (.mp4, .mkv, .mov, .webm, .avi)."
            )
            self._err_folder.setVisible(True)
            self._txt_folder.setStyleSheet("border: 1px solid #dc2626;")
            return

        self._detected_episodes = detected
        self._populate_episodes_table(detected)
        self._update_stepper(1)

    # -------------------------------------------------------------------------
    # Step 2: Discovered Episodes Checklist
    # -------------------------------------------------------------------------
    def _build_step2_episodes(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title = QLabel("Discovered Episodes")
        title.setObjectName("sectionTitle")
        self._lbl_episodes_summary = QLabel("Found 0 video files. Select episodes to localize:")
        self._lbl_episodes_summary.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(self._lbl_episodes_summary)

        # Bulk Selection Shortcuts
        selection_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_select_all.clicked.connect(lambda: self._set_all_episodes_checked(True))

        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_deselect_all.clicked.connect(lambda: self._set_all_episodes_checked(False))

        selection_row.addWidget(btn_select_all)
        selection_row.addWidget(btn_deselect_all)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)

        # Episodes Table
        self._table_episodes = QTableWidget()
        self._table_episodes.setColumnCount(4)
        self._table_episodes.setHorizontalHeaderLabels(
            ["Include", "Episode", "Filename", "File Size"]
        )
        self._table_episodes.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self._table_episodes.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table_episodes.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._table_episodes.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table_episodes.setColumnWidth(0, 65)
        self._table_episodes.verticalHeader().setVisible(False)
        self._table_episodes.setWordWrap(True)
        self._table_episodes.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._table_episodes.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_episodes.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_episodes.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table_episodes.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table_episodes.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table_episodes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table_episodes.verticalScrollBar().setSingleStep(16)
        layout.addWidget(self._table_episodes, stretch=1)

        self._err_episodes = QLabel("")
        self._err_episodes.setStyleSheet("color: #ef4444; font-size: 12px;")
        self._err_episodes.setVisible(False)
        layout.addWidget(self._err_episodes)

        # Actions Row
        actions_row = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._update_stepper(0))

        self._btn_step2_next = QPushButton("Next: Localization Setup →")
        self._btn_step2_next.setObjectName("primaryButton")
        self._btn_step2_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_step2_next.clicked.connect(self._on_step2_next)

        actions_row.addWidget(btn_back)
        actions_row.addStretch(1)
        actions_row.addWidget(self._btn_step2_next)
        layout.addLayout(actions_row)

        return page

    def _populate_episodes_table(self, episodes: list[tuple[int, Path]]) -> None:
        self._table_episodes.setRowCount(len(episodes))
        self._lbl_episodes_summary.setText(
            f"Found {len(episodes)} episode(s). Select episodes to localize:"
        )

        for row, (ep_num, path) in enumerate(episodes):
            # Checkbox item
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_item.setCheckState(Qt.CheckState.Checked)
            check_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table_episodes.setItem(row, 0, check_item)

            # Episode Tag
            ep_tag = f"EP {ep_num:02d}" if ep_num > 0 else "Special"
            tag_item = QTableWidgetItem(ep_tag)
            tag_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table_episodes.setItem(row, 1, tag_item)

            # Filename
            file_item = QTableWidgetItem(path.name)
            file_item.setToolTip(str(path))
            file_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table_episodes.setItem(row, 2, file_item)

            # File Size
            size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{(size_mb / 1024):.2f} GB"
            size_item = QTableWidgetItem(size_str)
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            self._table_episodes.setItem(row, 3, size_item)

        self._table_episodes.resizeRowsToContents()

    def _set_all_episodes_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self._table_episodes.rowCount()):
            item = self._table_episodes.item(row, 0)
            if item:
                item.setCheckState(state)

    def _on_step2_next(self) -> None:
        selected: list[tuple[int, Path]] = []
        for row in range(self._table_episodes.rowCount()):
            item = self._table_episodes.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                selected.append(self._detected_episodes[row])

        if not selected:
            self._err_episodes.setText("Please select at least 1 episode to proceed.")
            self._err_episodes.setVisible(True)
            return

        self._err_episodes.setVisible(False)
        self._selected_episodes = selected
        self._update_stepper(2)

    # -------------------------------------------------------------------------
    # Step 3: Localization & Series Glossary Settings
    # -------------------------------------------------------------------------
    def _build_step3_localization(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(14)

        # Scrollable form content so all options are comfortably accessible
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.verticalScrollBar().setSingleStep(16)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(14)

        title = QLabel("Localization Setup")
        title.setObjectName("sectionTitle")
        subtitle = QLabel(
            "Configure spoken language detection, workflow mode, and series terminology."
        )
        subtitle.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(4)

        # Source Language
        src_lbl = QLabel("Original Spoken Language")
        src_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(src_lbl)

        self._cmb_source_lang = QComboBox()
        self._cmb_source_lang.addItem("Auto-detect (Whisper audio probe)", None)
        self._cmb_source_lang.addItem("Chinese (Mandarin / Yue)", "zh")
        self._cmb_source_lang.addItem("Korean (ko)", "ko")
        self._cmb_source_lang.addItem("English (en)", "en")
        self._cmb_source_lang.addItem("Japanese (ja)", "ja")
        self._cmb_source_lang.addItem("Thai (th)", "th")
        self._cmb_source_lang.addItem("Vietnamese (vi)", "vi")
        layout.addWidget(self._cmb_source_lang)

        layout.addSpacing(6)

        # Target Language (Fixed to Khmer)
        tgt_lbl = QLabel("Target Language")
        tgt_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(tgt_lbl)

        txt_target = QLineEdit("Khmer (km) — Khmer Localization Engine")
        txt_target.setReadOnly(True)
        txt_target.setStyleSheet("color: #10b981; font-weight: 600;")
        layout.addWidget(txt_target)

        layout.addSpacing(6)

        # Workflow Mode
        mode_box = QGroupBox("Workflow Mode (Select for this dubbing run)")
        mode_box.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(8)

        self._rb_both = QRadioButton("Both (Voice Dubbing + Subtitles) — Synthesize audio dub & embed subtitles")
        self._rb_subtitles = QRadioButton("Subtitles Only — Generate timed Khmer subtitles (.srt, .ass, no audio dubbing)")
        self._rb_dubbing = QRadioButton("Voice Dubbing Only — Synthesize Khmer voice audio track (no subtitles)")

        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            self._rb_subtitles.setChecked(True)
        elif settings.output_mode == PipelineOutputMode.VOICE_DUBBING_ONLY:
            self._rb_dubbing.setChecked(True)
        else:
            self._rb_both.setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_both)
        mode_group.addButton(self._rb_subtitles)
        mode_group.addButton(self._rb_dubbing)

        mode_layout.addWidget(self._rb_both)
        mode_layout.addWidget(self._rb_subtitles)
        mode_layout.addWidget(self._rb_dubbing)
        layout.addWidget(mode_box)

        layout.addSpacing(6)

        # Series Glossary (Optional)
        glossary_lbl = QLabel("Series Glossary & Character Names (Optional)")
        glossary_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(glossary_lbl)

        glossary_helper = QLabel(
            "Paste JSON key-value pairs to enforce consistent translations "
            '(e.g. {"Lin Feng": "លីន ហ្វេង", "General": "មេទ័ព"}).'
        )
        glossary_helper.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(glossary_helper)

        self._txt_glossary = QLineEdit()
        self._txt_glossary.setPlaceholderText('{"Original Name": "Khmer Name"}')
        layout.addWidget(self._txt_glossary)

        self._err_glossary = QLabel("")
        self._err_glossary.setStyleSheet("color: #ef4444; font-size: 12px;")
        self._err_glossary.setVisible(False)
        layout.addWidget(self._err_glossary)

        # Status text for creation
        self._lbl_create_status = QLabel("")
        self._lbl_create_status.setStyleSheet("color: #3b82f6; font-size: 13px; font-weight: 500;")
        self._lbl_create_status.setVisible(False)
        layout.addWidget(self._lbl_create_status)

        layout.addStretch(1)

        scroll_area.setWidget(content_widget)
        page_layout.addWidget(scroll_area, stretch=1)

        # Actions Row (pinned at bottom)
        actions_row = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._update_stepper(1))

        self._btn_create = QPushButton("Start Project")
        self._btn_create.setObjectName("primaryButton")
        self._btn_create.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_create.clicked.connect(self._on_start_project)

        actions_row.addWidget(btn_back)
        actions_row.addStretch(1)
        actions_row.addWidget(self._btn_create)
        page_layout.addLayout(actions_row)

        return page

    def _on_start_project(self) -> None:
        """Validate inputs and dispatch project creation via AsyncBridge."""
        # Validate Glossary JSON if provided
        glossary_text = self._txt_glossary.text().strip()
        glossary_json: str | None = None
        if glossary_text:
            try:
                parsed = json.loads(glossary_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Glossary must be a JSON object (key-value dictionary).")
                glossary_json = json.dumps(parsed, ensure_ascii=False)
                self._err_glossary.setVisible(False)
            except Exception as exc:
                self._err_glossary.setText(f"Invalid JSON: {exc}")
                self._err_glossary.setVisible(True)
                return

        # Apply chosen workflow mode for this project / dubbing session
        if self._rb_subtitles.isChecked():
            settings.output_mode = PipelineOutputMode.SUBTITLES_ONLY
        elif self._rb_dubbing.isChecked():
            settings.output_mode = PipelineOutputMode.VOICE_DUBBING_ONLY
        else:
            settings.output_mode = PipelineOutputMode.BOTH

        # Collect parameters
        project_name = self._txt_name.text().strip()
        source_folder = self._source_dir
        source_language = self._cmb_source_lang.currentData()
        episodes = self._selected_episodes

        if not source_folder or not episodes:
            return

        # UI Loading State
        self._btn_create.setEnabled(False)
        self._btn_create.setText("Creating Project...")
        self._lbl_create_status.setText("Initializing project and indexing episodes...")
        self._lbl_create_status.setVisible(True)

        async def _create() -> int:
            async with get_session() as session:
                project = await create_project_from_folder(
                    session=session,
                    name=project_name,
                    source_folder=source_folder,
                    target_language="km",
                    source_language=source_language,
                    glossary_json=glossary_json,
                    episodes=episodes,
                )
                await session.commit()
                return project.id

        self._bridge.run_async(
            _create(),
            on_success=self._on_project_created_success,
            on_error=self._on_project_created_error,
        )

    def _on_project_created_success(self, project_id: int) -> None:
        """Main thread callback when project creation completes."""
        logger.info("Project created successfully with id={}", project_id)
        self._btn_create.setEnabled(True)
        self._btn_create.setText("Start Project")
        self._lbl_create_status.setVisible(False)
        self.project_created.emit(project_id)

    def _on_project_created_error(self, exc: Exception) -> None:
        """Main thread callback if project creation encounters an error."""
        logger.exception("Failed to create project: {}", exc)
        self._btn_create.setEnabled(True)
        user_msg = "Failed to create project. Please verify folder permissions and project name."
        if isinstance(exc, ValueError):
            user_msg = f"Invalid project input: {exc}"
        self._lbl_create_status.setText(user_msg)
        self._lbl_create_status.setStyleSheet("color: #ef4444; font-size: 13px;")
        self._lbl_create_status.setVisible(True)

    def reset(self) -> None:
        """Reset wizard fields to their initial clean state."""
        self._source_dir = None
        self._detected_episodes = []
        self._selected_episodes = []
        self._txt_folder.clear()
        self._txt_name.clear()
        self._txt_glossary.clear()
        self._err_folder.setVisible(False)
        self._err_name.setVisible(False)
        self._err_episodes.setVisible(False)
        self._err_glossary.setVisible(False)
        self._lbl_create_status.setVisible(False)
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            self._rb_subtitles.setChecked(True)
        elif settings.output_mode == PipelineOutputMode.VOICE_DUBBING_ONLY:
            self._rb_dubbing.setChecked(True)
        else:
            self._rb_both.setChecked(True)
        self._update_folder_preview(None)
        self._update_stepper(0)
