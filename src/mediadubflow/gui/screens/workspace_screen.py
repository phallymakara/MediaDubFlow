"""
Processing Workspace Screen — Main operational dashboard for MediaDubFlow.

Provides:
- Live episode queue table with real-time status badges, stage labels, and progress bars.
- Overall project progress metrics and aggregate KPIs (completed, active, pending, failed).
- Operational controls (Start All, Pause All, individual episode control).
- Activity ticker displaying latest engine and pipeline events.
- Thread-safe signal integration with JobManager.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config.settings import PipelineOutputMode, settings
from mediadubflow.database.repository import get_project
from mediadubflow.database.session import get_session
from mediadubflow.models.orm import Episode, EpisodeStatus, Project

if TYPE_CHECKING:
    from mediadubflow.core.job_manager import JobManager
    from mediadubflow.gui.async_bridge import AsyncBridge


class WorkspaceScreen(QWidget):
    """
    Main operational workspace monitoring and controlling episode processing.

    Emits:
        back_requested: Emitted when the user clicks '← Projects'.
        episode_selected: Emitted with episode_id when clicked for detailed inspection.
    """

    back_requested = Signal()
    episode_selected = Signal(int)

    def __init__(
        self,
        bridge: AsyncBridge,
        job_manager: JobManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._job_manager = job_manager

        self._project: Project | None = None
        self._episodes: list[Episode] = []
        # Mapping from episode_id to table row index
        self._episode_row_map: dict[int, int] = {}
        # Mapping from episode_id to widget elements for fast direct updates
        self._row_progress_bars: dict[int, QProgressBar] = {}
        self._row_stage_labels: dict[int, QLabel] = {}
        self._row_status_labels: dict[int, QLabel] = {}
        self._row_action_buttons: dict[int, QPushButton] = {}

        self._setup_ui()
        self._connect_job_signals()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 24, 32, 24)
        main_layout.setSpacing(16)

        # 1. Header & Controls Bar
        header_widget = self._build_header_controls()
        main_layout.addWidget(header_widget)

        # 1b. Audio & Dubbing Controls Bar (Mute Original Voice, Sound Volume, Speed Slider)
        audio_bar = self._build_audio_controls_bar()
        main_layout.addWidget(audio_bar)

        # 2. KPI Summary Bar & Overall Progress
        kpi_widget = self._build_kpi_progress_bar()
        main_layout.addWidget(kpi_widget)

        # 3. Live Episode Queue Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Episode", "Source Video", "Status", "Pipeline Progress", "Action"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 85)
        self._table.setColumnWidth(4, 110)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(52)
        self._table.setWordWrap(True)
        self._table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.cellDoubleClicked.connect(self._on_table_row_double_clicked)
        self._table.cellClicked.connect(self._on_table_cell_clicked)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        main_layout.addWidget(self._table, stretch=1)

        # 4. Activity Log Ticker Footer
        footer = self._build_activity_footer()
        main_layout.addWidget(footer)

    def _build_header_controls(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Back Navigation
        self._btn_back = QPushButton("← Projects")
        self._btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_back.clicked.connect(self.back_requested.emit)
        layout.addWidget(self._btn_back)

        layout.addSpacing(8)

        # Project Info
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self._lbl_project_title = QLabel("No Project Loaded")
        self._lbl_project_title.setWordWrap(True)
        self._lbl_project_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #0f172a;")
        self._lbl_project_meta = QLabel("")
        self._lbl_project_meta.setWordWrap(True)
        self._lbl_project_meta.setStyleSheet("font-size: 12px; color: #64748b;")
        title_box.addWidget(self._lbl_project_title)
        title_box.addWidget(self._lbl_project_meta)
        layout.addLayout(title_box)

        layout.addStretch(1)

        # Workflow Mode Selector (let user select mode when starting dubbing)
        mode_box = QHBoxLayout()
        mode_box.setSpacing(6)
        lbl_mode = QLabel("Workflow Mode:")
        lbl_mode.setStyleSheet("font-size: 12px; font-weight: 600; color: #475569;")
        self._cmb_workflow_mode = QComboBox()
        self._cmb_workflow_mode.setMinimumWidth(180)
        self._cmb_workflow_mode.addItem("Both (Dubbing + Subtitles)", PipelineOutputMode.BOTH)
        self._cmb_workflow_mode.addItem("Subtitles Only", PipelineOutputMode.SUBTITLES_ONLY)
        self._cmb_workflow_mode.addItem("Voice Dubbing Only", PipelineOutputMode.VOICE_DUBBING_ONLY)

        idx = self._cmb_workflow_mode.findData(settings.output_mode)
        if idx >= 0:
            self._cmb_workflow_mode.setCurrentIndex(idx)
        self._cmb_workflow_mode.currentIndexChanged.connect(self._on_workflow_mode_changed)

        mode_box.addWidget(lbl_mode)
        mode_box.addWidget(self._cmb_workflow_mode)
        layout.addLayout(mode_box)

        layout.addSpacing(6)

        # Global Control Buttons
        self._btn_start_all = QPushButton("Start All Episodes")
        self._btn_start_all.setObjectName("primaryButton")
        self._btn_start_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start_all.clicked.connect(self._on_start_all_clicked)

        self._btn_pause_all = QPushButton("Pause Queue")
        self._btn_pause_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_pause_all.clicked.connect(self._on_pause_all_clicked)

        self._btn_edit_speech = QPushButton("Edit Speech")
        self._btn_edit_speech.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_edit_speech.setEnabled(False)
        self._btn_edit_speech.clicked.connect(self._on_edit_speech_clicked)

        self._btn_redub_selected = QPushButton("Re-dub Selected")
        self._btn_redub_selected.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_redub_selected.setEnabled(False)
        self._btn_redub_selected.setToolTip("Re-synthesize dialogue at active speech speed and re-render the dubbed video")
        self._btn_redub_selected.clicked.connect(self._on_redub_selected_clicked)

        layout.addWidget(self._btn_start_all)
        layout.addWidget(self._btn_pause_all)
        layout.addWidget(self._btn_edit_speech)
        layout.addWidget(self._btn_redub_selected)

        return widget

    def _build_audio_controls_bar(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFixedHeight(56)

        widget = QWidget()
        widget.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 12px;"
        )
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(12)

        # 1. Mute Original Speaker Toggle
        self._chk_mute_original = QCheckBox("Mute Original Voice")
        self._chk_mute_original.setChecked(settings.mute_original_speaker)
        self._chk_mute_original.setStyleSheet("font-weight: 600; font-size: 12px; color: #0f172a;")
        self._chk_mute_original.toggled.connect(self._on_mute_original_toggled)
        layout.addWidget(self._chk_mute_original)

        layout.addSpacing(4)

        # 2. Sound Volume Control (Decrease, Horizontal Slider, Increase, Readout)
        vol_box = QHBoxLayout()
        vol_box.setSpacing(6)
        lbl_vol = QLabel("TTS Sound:")
        lbl_vol.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")

        self._btn_vol_down = QPushButton("-")
        self._btn_vol_down.setFixedSize(24, 24)
        self._btn_vol_down.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_vol_down.setToolTip("Decrease sound volume")
        self._btn_vol_down.clicked.connect(self._on_volume_decrease)

        self._slider_volume = QSlider(Qt.Orientation.Horizontal)
        self._slider_volume.setRange(0, 200)
        self._slider_volume.setSingleStep(5)
        self._slider_volume.setPageStep(10)
        self._slider_volume.setValue(int(settings.tts_volume * 100))
        self._slider_volume.setFixedWidth(110)
        self._slider_volume.setToolTip("Horizontal scroll or drag to adjust sound volume")
        self._slider_volume.valueChanged.connect(self._on_volume_slider_changed)

        self._btn_vol_up = QPushButton("+")
        self._btn_vol_up.setFixedSize(24, 24)
        self._btn_vol_up.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_vol_up.setToolTip("Increase sound volume")
        self._btn_vol_up.clicked.connect(self._on_volume_increase)

        self._lbl_vol_val = QLabel(f"{int(settings.tts_volume * 100)}%")
        self._lbl_vol_val.setFixedWidth(42)
        self._lbl_vol_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb;")

        vol_box.addWidget(lbl_vol)
        vol_box.addWidget(self._btn_vol_down)
        vol_box.addWidget(self._slider_volume)
        vol_box.addWidget(self._btn_vol_up)
        vol_box.addWidget(self._lbl_vol_val)
        layout.addLayout(vol_box)

        layout.addSpacing(4)

        # 3. Background Audio Volume Control
        bg_vol_box = QHBoxLayout()
        bg_vol_box.setSpacing(6)
        lbl_bg_vol = QLabel("Background Sound:")
        lbl_bg_vol.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")

        self._btn_bg_vol_down = QPushButton("-")
        self._btn_bg_vol_down.setFixedSize(24, 24)
        self._btn_bg_vol_down.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_bg_vol_down.setToolTip("Decrease background sound volume")
        self._btn_bg_vol_down.clicked.connect(self._on_bg_vol_decrease)

        self._slider_bg_volume = QSlider(Qt.Orientation.Horizontal)
        self._slider_bg_volume.setRange(0, 150)
        self._slider_bg_volume.setSingleStep(5)
        self._slider_bg_volume.setPageStep(10)
        self._slider_bg_volume.setValue(int(settings.original_audio_volume * 100))
        self._slider_bg_volume.setFixedWidth(100)
        self._slider_bg_volume.setToolTip("Horizontal scroll or drag to adjust background audio volume")
        self._slider_bg_volume.valueChanged.connect(self._on_bg_vol_slider_changed)

        self._btn_bg_vol_up = QPushButton("+")
        self._btn_bg_vol_up.setFixedSize(24, 24)
        self._btn_bg_vol_up.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_bg_vol_up.setToolTip("Increase background sound volume")
        self._btn_bg_vol_up.clicked.connect(self._on_bg_vol_increase)

        self._lbl_bg_vol_val = QLabel(f"{int(settings.original_audio_volume * 100)}%")
        self._lbl_bg_vol_val.setFixedWidth(40)
        self._lbl_bg_vol_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb;")

        bg_vol_box.addWidget(lbl_bg_vol)
        bg_vol_box.addWidget(self._btn_bg_vol_down)
        bg_vol_box.addWidget(self._slider_bg_volume)
        bg_vol_box.addWidget(self._btn_bg_vol_up)
        bg_vol_box.addWidget(self._lbl_bg_vol_val)
        layout.addLayout(bg_vol_box)

        layout.addSpacing(4)

        # 3. Horizontal Scroll Speed Control (0.50x to 2.00x)
        speed_box = QHBoxLayout()
        speed_box.setSpacing(6)
        lbl_speed = QLabel("Speech Speed:")
        lbl_speed.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")

        self._btn_speed_down = QPushButton("-")
        self._btn_speed_down.setFixedSize(24, 24)
        self._btn_speed_down.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_speed_down.setToolTip("Decrease speech speed")
        self._btn_speed_down.clicked.connect(self._on_speed_decrease)

        self._slider_speed = QSlider(Qt.Orientation.Horizontal)
        self._slider_speed.setRange(50, 200)
        self._slider_speed.setSingleStep(5)
        self._slider_speed.setPageStep(10)
        self._slider_speed.setValue(int(settings.tts_speed * 100))
        self._slider_speed.setFixedWidth(120)
        self._slider_speed.setToolTip("Horizontal scroll or drag to adjust speech speed (0.50x to 2.00x)")
        self._slider_speed.valueChanged.connect(self._on_speed_slider_changed)

        self._btn_speed_up = QPushButton("+")
        self._btn_speed_up.setFixedSize(24, 24)
        self._btn_speed_up.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_speed_up.setToolTip("Increase speech speed")
        self._btn_speed_up.clicked.connect(self._on_speed_increase)

        self._lbl_speed_val = QLabel(f"{settings.tts_speed:.2f}x")
        self._lbl_speed_val.setFixedWidth(44)
        self._lbl_speed_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb;")

        self._btn_reset_speed = QPushButton("1.5x")
        self._btn_reset_speed.setFixedHeight(24)
        self._btn_reset_speed.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_reset_speed.setToolTip("Reset to default 1.50x speed")
        self._btn_reset_speed.clicked.connect(lambda: self._slider_speed.setValue(150))

        speed_box.addWidget(lbl_speed)
        speed_box.addWidget(self._btn_speed_down)
        speed_box.addWidget(self._slider_speed)
        speed_box.addWidget(self._btn_speed_up)
        speed_box.addWidget(self._lbl_speed_val)
        speed_box.addWidget(self._btn_reset_speed)
        layout.addLayout(speed_box)

        layout.addStretch(1)

        scroll_area.setWidget(widget)
        return scroll_area

    def _on_mute_original_toggled(self, checked: bool) -> None:
        settings.mute_original_speaker = checked
        self._log_activity(
            "Original voice muted in dubbing." if checked else "Original voice blended into background."
        )

    def _on_volume_slider_changed(self, value: int) -> None:
        factor = round(value / 100.0, 2)
        settings.tts_volume = factor
        self._lbl_vol_val.setText(f"{value}%")

    def _on_volume_decrease(self) -> None:
        new_val = max(0, self._slider_volume.value() - 10)
        self._slider_volume.setValue(new_val)

    def _on_volume_increase(self) -> None:
        new_val = min(200, self._slider_volume.value() + 10)
        self._slider_volume.setValue(new_val)

    def _on_bg_vol_slider_changed(self, value: int) -> None:
        factor = round(value / 100.0, 2)
        settings.original_audio_volume = factor
        self._lbl_bg_vol_val.setText(f"{value}%")

    def _on_bg_vol_decrease(self) -> None:
        new_val = max(0, self._slider_bg_volume.value() - 10)
        self._slider_bg_volume.setValue(new_val)

    def _on_bg_vol_increase(self) -> None:
        new_val = min(150, self._slider_bg_volume.value() + 10)
        self._slider_bg_volume.setValue(new_val)

    def _on_speed_slider_changed(self, value: int) -> None:
        speed = round(value / 100.0, 2)
        settings.tts_speed = speed
        self._lbl_speed_val.setText(f"{speed:.2f}x")

    def _on_speed_decrease(self) -> None:
        new_val = max(50, self._slider_speed.value() - 5)
        self._slider_speed.setValue(new_val)

    def _on_speed_increase(self) -> None:
        new_val = min(200, self._slider_speed.value() + 5)
        self._slider_speed.setValue(new_val)

    def _build_kpi_progress_bar(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px;"
        )
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # KPI Metrics Row
        kpi_row = QHBoxLayout()
        self._lbl_kpi_summary = QLabel("0 Total  |  0 Done  |  0 In Progress  |  0 Pending")
        self._lbl_kpi_summary.setStyleSheet("font-weight: 600; font-size: 13px; color: #334155;")
        self._lbl_kpi_pct = QLabel("0% Completed")
        self._lbl_kpi_pct.setStyleSheet("font-weight: 700; font-size: 13px; color: #2563eb;")

        kpi_row.addWidget(self._lbl_kpi_summary)
        kpi_row.addStretch(1)
        kpi_row.addWidget(self._lbl_kpi_pct)
        layout.addLayout(kpi_row)

        # Overall Progress Bar
        self._overall_progress = QProgressBar()
        self._overall_progress.setRange(0, 100)
        self._overall_progress.setValue(0)
        layout.addWidget(self._overall_progress)

        return widget

    def _build_activity_footer(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(4, 2, 4, 2)

        lbl_lead = QLabel("Activity:")
        lbl_lead.setStyleSheet("font-weight: 600; color: #64748b; font-size: 12px;")

        self._lbl_activity_ticker = QLabel("Ready.")
        self._lbl_activity_ticker.setStyleSheet("color: #475569; font-size: 12px;")

        layout.addWidget(lbl_lead)
        layout.addWidget(self._lbl_activity_ticker, stretch=1)

        return footer

    # -------------------------------------------------------------------------
    # JobManager Signal Connections
    # -------------------------------------------------------------------------
    def _connect_job_signals(self) -> None:
        self._job_manager.progress_updated.connect(self._on_job_progress_updated)
        self._job_manager.status_changed.connect(self._on_job_status_changed)
        self._job_manager.episode_failed.connect(self._on_job_episode_failed)
        self._job_manager.episode_completed.connect(self._on_job_episode_completed)

    # -------------------------------------------------------------------------
    # Project Loading
    # -------------------------------------------------------------------------
    def load_project(self, project_id: int) -> None:
        """Fetch project details and episodes asynchronously from SQLite."""
        self._log_activity(f"Loading project #{project_id}...")

        async def _fetch() -> Project | None:
            async with get_session() as session:
                return await get_project(session, project_id)

        self._bridge.run_async(
            _fetch(),
            on_success=self._on_project_loaded,
            on_error=lambda exc: logger.error("Failed to load project #{}: {}", project_id, exc),
        )

    def _on_project_loaded(self, project: Project | None) -> None:
        if not project:
            self._lbl_project_title.setText("Project not found")
            return

        self._project = project
        self._episodes = list(project.episodes) if project.episodes else []
        self._lbl_project_title.setText(project.name)
        self._lbl_project_meta.setText(
            f"Target: Khmer (km)  |  Path: {project.source_folder}  |  Episodes: {len(self._episodes)}"
        )

        mode_idx = self._cmb_workflow_mode.findData(settings.output_mode)
        if mode_idx >= 0:
            self._cmb_workflow_mode.blockSignals(True)
            self._cmb_workflow_mode.setCurrentIndex(mode_idx)
            self._cmb_workflow_mode.blockSignals(False)

        self._populate_table()
        self._recalculate_kpis()
        self._log_activity(f"Loaded '{project.name}' with {len(self._episodes)} episode(s).")

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._episodes))
        self._episode_row_map.clear()
        self._row_progress_bars.clear()
        self._row_stage_labels.clear()
        self._row_status_labels.clear()
        self._row_action_buttons.clear()

        for row, ep in enumerate(self._episodes):
            self._episode_row_map[ep.id] = row
            self._table.setRowHeight(row, 52)

            # Col 0: Episode Tag
            tag = f"EP {ep.episode_number:02d}" if ep.episode_number > 0 else "Special"
            tag_item = QTableWidgetItem(tag)
            tag_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 0, tag_item)

            # Col 1: Filename
            filename = (
                Path(ep.source_file).name if ep.source_file else f"Episode {ep.episode_number}"
            )
            file_item = QTableWidgetItem(filename)
            file_item.setToolTip(ep.source_file or filename)
            file_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._table.setItem(row, 1, file_item)

            # Col 2: Status Badge Widget
            status_lbl = self._create_status_badge(ep.status)
            self._row_status_labels[ep.id] = status_lbl
            self._table.setCellWidget(row, 2, self._wrap_cell(status_lbl))

            # Col 3: Progress & Stage
            progress_widget, prog_bar, stage_lbl = self._create_progress_cell(ep)
            self._row_progress_bars[ep.id] = prog_bar
            self._row_stage_labels[ep.id] = stage_lbl
            self._table.setCellWidget(row, 3, progress_widget)

            # Col 4: Action Button
            action_btn = QPushButton()
            action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._update_action_button(action_btn, ep)
            self._row_action_buttons[ep.id] = action_btn
            self._table.setCellWidget(row, 4, self._wrap_cell(action_btn))

    def _wrap_cell(self, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(widget)
        return container

    def _create_status_badge(self, status: EpisodeStatus) -> QLabel:
        lbl = QLabel(status.value.replace("_", " ").title())
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setProperty("class", "statusBadge")
        self._apply_badge_style(lbl, status)
        return lbl

    @staticmethod
    def _apply_badge_style(lbl: QLabel, status: EpisodeStatus) -> None:
        if status == EpisodeStatus.DONE:
            lbl.setStyleSheet(
                "border: 1px solid #16a34a; color: #16a34a; font-weight: 600; font-size: 11px; padding: 2px 6px; border-radius: 4px;"
            )
        elif status == EpisodeStatus.FAILED:
            lbl.setStyleSheet(
                "border: 1px solid #dc2626; color: #dc2626; font-weight: 600; font-size: 11px; padding: 2px 6px; border-radius: 4px;"
            )
        elif status == EpisodeStatus.PENDING:
            lbl.setStyleSheet(
                "border: 1px solid #cbd5e1; color: #64748b; font-weight: 500; font-size: 11px; padding: 2px 6px; border-radius: 4px;"
            )
        else:
            lbl.setStyleSheet(
                "border: 1px solid #2563eb; color: #2563eb; font-weight: 600; font-size: 11px; padding: 2px 6px; border-radius: 4px;"
            )

    def _create_progress_cell(self, ep: Episode) -> tuple[QWidget, QProgressBar, QLabel]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        stage_name = self._status_to_stage_name(ep.status)
        stage_lbl = QLabel(stage_name)
        stage_lbl.setStyleSheet("font-size: 11px; color: #475569;")

        prog_bar = QProgressBar()
        prog_bar.setRange(0, 100)
        prog_val = 100 if ep.status == EpisodeStatus.DONE else 0
        prog_bar.setValue(prog_val)

        layout.addWidget(stage_lbl)
        layout.addWidget(prog_bar)
        return widget, prog_bar, stage_lbl

    def _update_action_button(self, btn: QPushButton, ep: Episode) -> None:
        if btn.receivers(btn.clicked) > 0:
            try:
                btn.clicked.disconnect()
            except RuntimeError:
                pass

        if ep.status == EpisodeStatus.DONE:
            btn.setText("Review")
            btn.setStyleSheet("")
            btn.clicked.connect(lambda _, ep_id=ep.id: self.episode_selected.emit(ep_id))
        elif ep.translation_path or ep.status == EpisodeStatus.REVIEW:
            btn.setText("Edit Speech")
            btn.setStyleSheet("")
            btn.clicked.connect(lambda _, ep_id=ep.id: self.episode_selected.emit(ep_id))
        elif ep.status == EpisodeStatus.FAILED:
            btn.setText("Retry")
            btn.setStyleSheet("color: #dc2626; font-weight: 600;")
            btn.clicked.connect(lambda _, ep_id=ep.id: self._enqueue_episode(ep_id))
        elif ep.status == EpisodeStatus.PENDING:
            btn.setText("Start")
            btn.setStyleSheet("")
            btn.clicked.connect(lambda _, ep_id=ep.id: self._enqueue_episode(ep_id))
        else:
            btn.setText("Active")
            btn.setEnabled(False)

    @staticmethod
    def _status_to_stage_name(status: EpisodeStatus) -> str:
        if status == EpisodeStatus.DONE:
            return (
                "Complete (Dubbed Video ready)"
                if settings.output_mode != PipelineOutputMode.SUBTITLES_ONLY
                else "Complete (Subtitles ready)"
            )
        if status == EpisodeStatus.FAILED:
            return "Failed"
        if status == EpisodeStatus.PENDING:
            return "Queued"
        return status.value.replace("_", " ").title()

    # -------------------------------------------------------------------------
    # Job Control Actions
    # -------------------------------------------------------------------------
    def _on_workflow_mode_changed(self) -> None:
        """Update active pipeline workflow mode when user changes the dropdown."""
        mode = self._cmb_workflow_mode.currentData()
        if mode:
            settings.output_mode = mode
            self._log_activity(f"Workflow mode set to: {mode.value}")

    def _enqueue_episode(self, episode_id: int) -> None:
        # Ensure active workflow mode from dropdown is applied
        mode = self._cmb_workflow_mode.currentData()
        if mode:
            settings.output_mode = mode

        self._job_manager.enqueue(episode_id)
        if episode_id in self._row_status_labels:
            lbl = self._row_status_labels[episode_id]
            lbl.setText("Queued")
            self._apply_badge_style(lbl, EpisodeStatus.PENDING)
        if episode_id in self._row_action_buttons:
            self._row_action_buttons[episode_id].setEnabled(False)
            self._row_action_buttons[episode_id].setText("Queued")
        self._log_activity(f"Episode #{episode_id} enqueued for processing.")

    def _on_start_all_clicked(self) -> None:
        """Enqueue all pending or failed episodes into JobManager."""
        mode = self._cmb_workflow_mode.currentData()
        if mode:
            settings.output_mode = mode

        count = 0
        for ep in self._episodes:
            if ep.status in (EpisodeStatus.PENDING, EpisodeStatus.FAILED):
                self._enqueue_episode(ep.id)
                count += 1
        self._log_activity(f"Started queue: {count} episode(s) added.")

    def _on_pause_all_clicked(self) -> None:
        """Stop worker dispatch loop."""
        self._job_manager.stop()
        self._log_activity("Processing paused. Active stages will finish.")

    # -------------------------------------------------------------------------
    # Slot Callbacks from JobManager
    # -------------------------------------------------------------------------
    def _on_job_progress_updated(self, episode_id: int, stage_name: str, pct: int) -> None:
        if episode_id in self._row_progress_bars:
            self._row_progress_bars[episode_id].setValue(pct)
        if episode_id in self._row_stage_labels:
            if f"({pct}%)" in stage_name or "%" in stage_name:
                self._row_stage_labels[episode_id].setText(stage_name)
            else:
                self._row_stage_labels[episode_id].setText(f"{stage_name} ({pct}%)")

        self._recalculate_kpis()
        self._log_activity(f"EP #{episode_id}: {stage_name} at {pct}%")

    def _on_job_status_changed(self, episode_id: int, status_str: str) -> None:
        try:
            status = EpisodeStatus(status_str)
        except ValueError:
            status = EpisodeStatus.PENDING

        if episode_id in self._row_status_labels:
            lbl = self._row_status_labels[episode_id]
            lbl.setText(status.value.replace("_", " ").title())
            self._apply_badge_style(lbl, status)

        # Update cached episode status
        for ep in self._episodes:
            if ep.id == episode_id:
                ep.status = status
                break

        self._recalculate_kpis()
        self._log_activity(f"EP #{episode_id}: Status changed to {status.value}")

    def _on_job_episode_failed(self, episode_id: int, error_msg: str) -> None:
        if episode_id in self._row_status_labels:
            lbl = self._row_status_labels[episode_id]
            lbl.setText("Failed")
            self._apply_badge_style(lbl, EpisodeStatus.FAILED)

        if episode_id in self._row_stage_labels:
            self._row_stage_labels[episode_id].setText(f"Error: {error_msg[:40]}...")

        if episode_id in self._row_action_buttons:
            btn = self._row_action_buttons[episode_id]
            btn.setEnabled(True)
            btn.setText("Retry")
            btn.setStyleSheet("color: #dc2626; font-weight: 600;")
            try:
                btn.clicked.disconnect()
            except RuntimeError:
                pass
            btn.clicked.connect(lambda _, ep_id=episode_id: self._enqueue_episode(ep_id))

        self._recalculate_kpis()
        self._log_activity(f"EP #{episode_id} failed: {error_msg}")

    def _on_job_episode_completed(self, episode_id: int) -> None:
        if episode_id in self._row_status_labels:
            lbl = self._row_status_labels[episode_id]
            lbl.setText("Done")
            self._apply_badge_style(lbl, EpisodeStatus.DONE)

        if episode_id in self._row_progress_bars:
            self._row_progress_bars[episode_id].setValue(100)

        if episode_id in self._row_stage_labels:
            msg = (
                "Complete (Dubbed Video ready)"
                if settings.output_mode != PipelineOutputMode.SUBTITLES_ONLY
                else "Complete (Subtitles ready)"
            )
            self._row_stage_labels[episode_id].setText(msg)

        if episode_id in self._row_action_buttons:
            btn = self._row_action_buttons[episode_id]
            btn.setEnabled(True)
            btn.setText("Review")
            btn.setStyleSheet("")
            try:
                btn.clicked.disconnect()
            except RuntimeError:
                pass
            btn.clicked.connect(lambda _, ep_id=episode_id: self.episode_selected.emit(ep_id))

        for ep in self._episodes:
            if ep.id == episode_id:
                ep.status = EpisodeStatus.DONE
                break

        self._recalculate_kpis()
        self._log_activity(f"EP #{episode_id} completed successfully!")

    # -------------------------------------------------------------------------
    # KPIs & UI Helpers
    # -------------------------------------------------------------------------
    def _recalculate_kpis(self) -> None:
        if not self._episodes:
            self._overall_progress.setValue(0)
            self._lbl_kpi_summary.setText("0 Total  |  0 Done  |  0 In Progress  |  0 Pending")
            self._lbl_kpi_pct.setText("0% Completed")
            return

        total = len(self._episodes)
        done = sum(1 for ep in self._episodes if ep.status == EpisodeStatus.DONE)
        failed = sum(1 for ep in self._episodes if ep.status == EpisodeStatus.FAILED)
        pending = sum(1 for ep in self._episodes if ep.status == EpisodeStatus.PENDING)
        in_prog = total - (done + failed + pending)

        pct = int((done / total) * 100) if total > 0 else 0
        self._overall_progress.setValue(pct)
        self._lbl_kpi_pct.setText(f"{pct}% Completed")

        fail_text = f"  |  {failed} Failed" if failed > 0 else ""
        self._lbl_kpi_summary.setText(
            f"{total} Total  |  {done} Done  |  {in_prog} In Progress  |  {pending} Pending{fail_text}"
        )

    def _on_table_cell_clicked(self, row: int, col: int) -> None:
        """Handle single-click on episode table rows."""
        if 0 <= row < len(self._episodes):
            ep_id = self._episodes[row].id
            self._update_selected_episode_ui(ep_id)
            # If user clicked any content column other than Action button, open editor
            if col != 4:
                self.episode_selected.emit(ep_id)

    def _on_table_selection_changed(self) -> None:
        """Update header edit button when table selection changes."""
        selected_rows = self._table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self._episodes):
                ep_id = self._episodes[row].id
                self._update_selected_episode_ui(ep_id)
                return
        self._btn_edit_speech.setEnabled(False)
        self._btn_edit_speech.setText("Edit Speech")
        self._btn_redub_selected.setEnabled(False)
        self._btn_redub_selected.setText("Re-dub Selected")

    def _update_selected_episode_ui(self, episode_id: int) -> None:
        for ep in self._episodes:
            if ep.id == episode_id:
                ep_tag = f"EP {ep.episode_number:02d}" if ep.episode_number > 0 else f"#{ep.id}"
                self._btn_edit_speech.setEnabled(True)
                self._btn_edit_speech.setText(f"Edit Speech ({ep_tag})")
                can_redub = bool(
                    ep.status in (EpisodeStatus.DONE, EpisodeStatus.REVIEW)
                    or ep.translation_path
                )
                self._btn_redub_selected.setEnabled(can_redub)
                self._btn_redub_selected.setText(f"Re-dub ({ep_tag})")
                break

    def _on_redub_selected_clicked(self) -> None:
        """Trigger rapid re-dubbing with active speech speed for currently selected episode."""
        selected_rows = self._table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self._episodes):
                ep = self._episodes[row]
                self._job_manager.redub_episode(ep.id, settings.tts_speed)
                if ep.id in self._row_status_labels:
                    lbl = self._row_status_labels[ep.id]
                    lbl.setText("Queued")
                    self._apply_badge_style(lbl, EpisodeStatus.PENDING)
                if ep.id in self._row_progress_bars:
                    self._row_progress_bars[ep.id].setValue(0)
                if ep.id in self._row_action_buttons:
                    self._row_action_buttons[ep.id].setEnabled(False)
                    self._row_action_buttons[ep.id].setText("Queued")
                self._log_activity(f"Episode #{ep.id} queued to re-dub at {settings.tts_speed:.2f}x speed.")

    def _on_edit_speech_clicked(self) -> None:
        """Trigger episode inspector & editor for currently selected episode."""
        selected_rows = self._table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self._episodes):
                ep_id = self._episodes[row].id
                self.episode_selected.emit(ep_id)

    def _on_table_row_double_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._episodes):
            ep_id = self._episodes[row].id
            self.episode_selected.emit(ep_id)

    def _log_activity(self, message: str) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{now_str}] {message}"
        self._lbl_activity_ticker.setText(full_msg)
        self._lbl_activity_ticker.setToolTip(full_msg)
