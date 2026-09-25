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
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.database.session import get_session
from mediadubflow.services.project_service import create_project_from_folder
from mediadubflow.utils.episode_detector import detect_episodes

if TYPE_CHECKING:
    from mediadubflow.gui.async_bridge import AsyncBridge


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
        main_layout.setContentsMargins(40, 32, 40, 32)
        main_layout.setSpacing(20)

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
    # Stepper Indicator
    # -------------------------------------------------------------------------
    def _build_stepper(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(12)

        self._step_labels: list[QLabel] = []
        steps = ["1. Media Folder", "2. Episodes Review", "3. Localization Setup"]

        for i, title in enumerate(steps):
            lbl = QLabel(title)
            lbl.setStyleSheet(
                "font-size: 13px; font-weight: 600; color: #64748b; padding: 4px 8px;"
            )
            self._step_labels.append(lbl)
            layout.addWidget(lbl)

            if i < len(steps) - 1:
                divider = QLabel("───")
                divider.setStyleSheet("color: #334155; font-size: 12px;")
                layout.addWidget(divider)

        layout.addStretch(1)
        return widget

    def _update_stepper(self, active_index: int) -> None:
        self._step_stack.setCurrentIndex(active_index)
        for i, lbl in enumerate(self._step_labels):
            if i == active_index:
                lbl.setStyleSheet(
                    "font-size: 13px; font-weight: 700; color: #3b82f6; "
                    "border-bottom: 2px solid #3b82f6; padding: 4px 8px;"
                )
            elif i < active_index:
                lbl.setStyleSheet(
                    "font-size: 13px; font-weight: 600; color: #10b981; padding: 4px 8px;"
                )
            else:
                lbl.setStyleSheet(
                    "font-size: 13px; font-weight: 500; color: #64748b; padding: 4px 8px;"
                )

    # -------------------------------------------------------------------------
    # Step 1: Media Ingestion & Project Setup
    # -------------------------------------------------------------------------
    def _build_step1_media(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title = QLabel("Select Media Folder")
        title.setObjectName("sectionTitle")
        subtitle = QLabel("Choose the directory containing your drama or movie video files.")
        subtitle.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        # Source Folder Picker
        folder_lbl = QLabel("Media Folder")
        folder_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(folder_lbl)

        folder_row = QHBoxLayout()
        self._txt_folder = QLineEdit()
        self._txt_folder.setPlaceholderText(
            "Select folder containing video files (.mp4, .mkv, .mov, .webm)"
        )
        self._txt_folder.setReadOnly(True)

        btn_browse = QPushButton("Browse...")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._on_browse_folder)

        folder_row.addWidget(self._txt_folder, stretch=1)
        folder_row.addWidget(btn_browse)
        layout.addLayout(folder_row)

        self._err_folder = QLabel("")
        self._err_folder.setStyleSheet("color: #ef4444; font-size: 12px; margin-top: 2px;")
        self._err_folder.setVisible(False)
        layout.addWidget(self._err_folder)

        layout.addSpacing(12)

        # Project Name
        name_lbl = QLabel("Project Title")
        name_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(name_lbl)

        self._txt_name = QLineEdit()
        self._txt_name.setPlaceholderText("Enter series or drama title")
        self._txt_name.textChanged.connect(self._clear_name_error)
        layout.addWidget(self._txt_name)

        self._err_name = QLabel("")
        self._err_name.setStyleSheet("color: #ef4444; font-size: 12px; margin-top: 2px;")
        self._err_name.setVisible(False)
        layout.addWidget(self._err_name)

        layout.addStretch(1)

        # Actions Row
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
        layout.addLayout(actions_row)

        return page

    def _on_browse_folder(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Drama/Movie Folder",
            str(Path.home()),
        )
        if not selected_dir:
            return

        folder_path = Path(selected_dir)
        self._txt_folder.setText(str(folder_path))
        self._source_dir = folder_path

        # Clear inline folder error
        self._err_folder.setVisible(False)
        self._txt_folder.setStyleSheet("")

        # Auto-populate project name if empty
        if not self._txt_name.text().strip():
            self._txt_name.setText(folder_path.name.replace("_", " ").title())
            self._clear_name_error()

    def _clear_name_error(self) -> None:
        self._err_name.setVisible(False)
        self._txt_name.setStyleSheet("")

    def _on_step1_next(self) -> None:
        # Validate Project Name
        name = self._txt_name.text().strip()
        if not name:
            self._err_name.setText("Project title is required.")
            self._err_name.setVisible(True)
            self._txt_name.setStyleSheet("border: 1px solid #ef4444;")
            return

        # Validate Folder
        if not self._source_dir or not self._source_dir.exists():
            self._err_folder.setText("Please select a valid media folder.")
            self._err_folder.setVisible(True)
            self._txt_folder.setStyleSheet("border: 1px solid #ef4444;")
            return

        # Detect episodes
        try:
            detected = detect_episodes(self._source_dir)
        except Exception as exc:
            logger.exception("Failed to scan directory for episodes: {}", exc)
            self._err_folder.setText(f"Unable to read folder: {exc}")
            self._err_folder.setVisible(True)
            self._txt_folder.setStyleSheet("border: 1px solid #ef4444;")
            return

        if not detected:
            self._err_folder.setText(
                "No supported video files found (.mp4, .mkv, .mov, .webm, .avi)."
            )
            self._err_folder.setVisible(True)
            self._txt_folder.setStyleSheet("border: 1px solid #ef4444;")
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
        self._table_episodes.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_episodes.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
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
            file_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table_episodes.setItem(row, 2, file_item)

            # File Size
            size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{(size_mb / 1024):.2f} GB"
            size_item = QTableWidgetItem(size_str)
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            self._table_episodes.setItem(row, 3, size_item)

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
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

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

        layout.addSpacing(8)

        # Target Language (Fixed to Khmer)
        tgt_lbl = QLabel("Target Language")
        tgt_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(tgt_lbl)

        txt_target = QLineEdit("Khmer (km) — Khmer Localization Engine")
        txt_target.setReadOnly(True)
        txt_target.setStyleSheet("color: #10b981; font-weight: 600;")
        layout.addWidget(txt_target)

        layout.addSpacing(8)

        # Workflow Mode
        mode_box = QGroupBox("Workflow Mode")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(6)

        self._rb_subtitles = QRadioButton("Subtitles Only (.srt, .ass)")
        self._rb_subtitles.setChecked(True)
        self._rb_dubbing = QRadioButton("Full Dubbing (Synthesized Audio + Subtitles)")

        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_subtitles)
        mode_group.addButton(self._rb_dubbing)

        mode_layout.addWidget(self._rb_subtitles)
        mode_layout.addWidget(self._rb_dubbing)
        layout.addWidget(mode_box)

        layout.addSpacing(8)

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

        # Actions Row
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
        layout.addLayout(actions_row)

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
        self._btn_create.setText("Start Project")
        self._lbl_create_status.setText(f"Error creating project: {exc}")
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
        self._update_stepper(0)
