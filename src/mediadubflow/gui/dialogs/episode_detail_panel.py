"""
Episode Detail & Speech Translation CRUD Editor Panel.

Provides:
- Comprehensive episode metadata and pipeline status inspection.
- Full CRUD editor for AI-translated speech segments (dialogue timing, original text, Khmer translation).
- Real-time search and filtering across original speech and translated Khmer lines.
- Automatic synchronization of .srt and .ass subtitle files upon saving translation edits.
- Preview of compiled subtitles and direct video playback.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config.settings import settings
from mediadubflow.models.orm import Episode, EpisodeStatus
from mediadubflow.utils.subtitles import create_ass_file, create_srt_file


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.mmm string (e.g. 01:23.450)."""
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:06.3f}"


def parse_timestamp(text: str) -> float | None:
    """Parse MM:SS.mmm or numeric seconds into float seconds."""
    text = text.strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                m = float(parts[0])
                s = float(parts[1])
                return max(0.0, m * 60.0 + s)
            except ValueError:
                return None
    try:
        val = float(text)
        return max(0.0, val)
    except ValueError:
        return None


class SpeechSegmentEditDialog(QDialog):
    """Modal dialog for creating or updating a single translated speech segment."""

    def __init__(
        self,
        segment: dict[str, Any] | None = None,
        default_start: float = 0.0,
        default_end: float = 2.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._segment = segment
        is_edit = segment is not None
        self.setWindowTitle("Edit Speech Line" if is_edit else "Add New Speech Line")
        self.setMinimumSize(540, 420)
        self.resize(580, 460)

        self._setup_ui(is_edit, default_start, default_end)

    def _setup_ui(
        self,
        is_edit: bool,
        default_start: float,
        default_end: float,
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # Header Title
        title_text = (
            f"Edit Speech Line #{self._segment.get('id', 1)}"
            if is_edit and self._segment
            else "New Speech Dialogue Line"
        )
        lbl_title = QLabel(title_text)
        lbl_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        layout.addWidget(lbl_title)

        # Timestamps Row
        time_row = QHBoxLayout()
        time_row.setSpacing(12)

        start_val = self._segment.get("start", default_start) if self._segment else default_start
        end_val = self._segment.get("end", default_end) if self._segment else default_end

        # Start Time
        start_box = QVBoxLayout()
        start_box.setSpacing(4)
        lbl_start = QLabel("Start Time (MM:SS.mmm or seconds):")
        lbl_start.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self._txt_start = QLineEdit(format_timestamp(start_val))
        start_box.addWidget(lbl_start)
        start_box.addWidget(self._txt_start)
        time_row.addLayout(start_box)

        # End Time
        end_box = QVBoxLayout()
        end_box.setSpacing(4)
        lbl_end = QLabel("End Time (MM:SS.mmm or seconds):")
        lbl_end.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self._txt_end = QLineEdit(format_timestamp(end_val))
        end_box.addWidget(lbl_end)
        end_box.addWidget(self._txt_end)
        time_row.addLayout(end_box)

        layout.addLayout(time_row)

        # Original Dialogue
        lbl_orig = QLabel("Original Dialogue (Transcribed Speech):")
        lbl_orig.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        self._txt_orig = QPlainTextEdit()
        self._txt_orig.setMinimumHeight(75)
        self._txt_orig.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._txt_orig.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        orig_text = self._segment.get("text", "") if self._segment else ""
        self._txt_orig.setPlainText(orig_text)
        layout.addWidget(lbl_orig)
        layout.addWidget(self._txt_orig, stretch=1)

        # Translated Khmer Speech
        lbl_trans = QLabel("Khmer Translation (បកប្រែជាភាសាខ្មែរ):")
        lbl_trans.setStyleSheet("font-weight: 600; font-size: 12px; color: #0f172a;")
        self._txt_trans = QPlainTextEdit()
        self._txt_trans.setMinimumHeight(95)
        self._txt_trans.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._txt_trans.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        trans_text = (
            self._segment.get("translated_text", "")
            if self._segment
            else ""
        )
        self._txt_trans.setPlainText(trans_text)
        layout.addWidget(lbl_trans)
        layout.addWidget(self._txt_trans, stretch=1)

        # Error text area (hidden by default)
        self._lbl_error = QLabel("")
        self._lbl_error.setStyleSheet("color: #dc2626; font-size: 12px; font-weight: 500;")
        self._lbl_error.setVisible(False)
        layout.addWidget(self._lbl_error)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        self._btn_save = QPushButton("Save Speech Line")
        self._btn_save.setObjectName("primaryButton")
        self._btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_save.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(self._btn_save)

        layout.addLayout(btn_layout)

    def _on_save_clicked(self) -> None:
        start_sec = parse_timestamp(self._txt_start.text())
        if start_sec is None:
            self._show_error("Invalid start time format. Use MM:SS.mmm (e.g. 01:23.500) or numeric seconds.")
            return

        end_sec = parse_timestamp(self._txt_end.text())
        if end_sec is None:
            self._show_error("Invalid end time format. Use MM:SS.mmm (e.g. 01:25.000) or numeric seconds.")
            return

        if end_sec <= start_sec:
            self._show_error("End time must be greater than start time.")
            return

        trans_text = self._txt_trans.toPlainText().strip()
        if not trans_text:
            self._show_error("Khmer translation cannot be empty.")
            return

        self._lbl_error.setVisible(False)
        self.accept()

    def _show_error(self, message: str) -> None:
        self._lbl_error.setText(message)
        self._lbl_error.setVisible(True)

    def get_segment_data(self) -> dict[str, Any]:
        """Return updated segment dictionary."""
        start_sec = parse_timestamp(self._txt_start.text()) or 0.0
        end_sec = parse_timestamp(self._txt_end.text()) or (start_sec + 1.0)
        orig_text = self._txt_orig.toPlainText().strip()
        trans_text = self._txt_trans.toPlainText().strip()

        seg_id = self._segment.get("id", 1) if self._segment else 1
        speaker = self._segment.get("speaker") if self._segment else None

        result: dict[str, Any] = {
            "id": seg_id,
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "text": orig_text,
            "translated_text": trans_text,
        }
        if speaker is not None:
            result["speaker"] = speaker
        return result


class EpisodeDetailPanel(QDialog):
    """
    Inspector and Speech CRUD Editor for reviewing and editing translated speech.

    Allows operators to:
    - Inspect episode status, detected language, and file outputs.
    - View and filter all AI-translated dialogue segments.
    - Create, Edit, and Delete dialogue lines with full timestamp control.
    - Synchronize updated dialogue directly back to JSON and subtitle files (.srt, .ass).
    - Play dubbed videos and open containing media folders.
    """

    def __init__(
        self,
        episode: Episode,
        parent: QWidget | None = None,
        job_manager: Any = None,
    ) -> None:
        super().__init__(parent)
        self.episode = episode
        self._job_manager = job_manager or getattr(parent, "_job_manager", None)
        self.setWindowTitle(f"Episode {episode.episode_number:02d} Review & Speech Editor — MediaDubFlow")
        self.setMinimumSize(960, 620)
        self.resize(1080, 700)

        self._segments: list[dict[str, Any]] = []
        self._filtered_indices: list[int] = []
        self._has_unsaved_changes = False

        self._setup_ui()
        self._load_segments()
        self._load_episode_data()

        if self._job_manager:
            self._job_manager.progress_updated.connect(self._on_job_progress)
            self._job_manager.episode_completed.connect(self._on_job_completed)
            self._job_manager.episode_failed.connect(self._on_job_failed)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # 1. Header Bar
        header_layout = QHBoxLayout()
        title = QLabel(f"Episode {self.episode.episode_number:02d} Inspector & Speech Editor")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #0f172a;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        self._status_badge = QLabel(self.episode.status.value.replace("_", " ").title())
        self._status_badge.setStyleSheet(
            "font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; "
            "border: 1px solid #2563eb; color: #2563eb;"
        )
        header_layout.addWidget(self._status_badge)
        main_layout.addLayout(header_layout)

        # 2. Main Tabbed Container
        self._tabs = QTabWidget()

        # Tab 0: Speech Translation CRUD Editor
        tab_editor = self._build_speech_editor_tab()
        self._tabs.addTab(tab_editor, "Translated Speech (AI Dubbing)")

        # Tab 1: Subtitle Files & Video Inspection
        tab_preview = self._build_preview_tab()
        self._tabs.addTab(tab_preview, "Subtitle Files & Video Output")

        main_layout.addWidget(self._tabs, stretch=1)

        # 3. Dedicated Speed Adjustment & Re-dub Toolbar
        speed_panel = QFrame()
        speed_panel.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 6px 12px;"
        )
        speed_layout = QHBoxLayout(speed_panel)
        speed_layout.setContentsMargins(10, 6, 10, 6)
        speed_layout.setSpacing(8)

        lbl_speed = QLabel("Dubbing Speed:")
        lbl_speed.setStyleSheet("font-weight: 600; font-size: 12px; color: #334155;")

        self._btn_speed_down = QPushButton("-")
        self._btn_speed_down.setFixedSize(24, 24)
        self._btn_speed_down.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_speed_down.setToolTip("Decrease speech speed")
        self._btn_speed_down.clicked.connect(self._on_speed_decrease_clicked)

        self._slider_speed = QSlider(Qt.Orientation.Horizontal)
        self._slider_speed.setRange(50, 200)
        self._slider_speed.setSingleStep(5)
        self._slider_speed.setPageStep(10)
        self._slider_speed.setValue(int(settings.tts_speed * 100))
        self._slider_speed.setFixedWidth(130)
        self._slider_speed.setToolTip("Horizontal scroll or drag to adjust speech speed (0.50x to 2.00x)")
        self._slider_speed.valueChanged.connect(self._on_speed_slider_changed)

        self._btn_speed_up = QPushButton("+")
        self._btn_speed_up.setFixedSize(24, 24)
        self._btn_speed_up.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_speed_up.setToolTip("Increase speech speed")
        self._btn_speed_up.clicked.connect(self._on_speed_increase_clicked)

        self._lbl_speed_val = QLabel(f"{settings.tts_speed:.2f}x")
        self._lbl_speed_val.setFixedWidth(46)
        self._lbl_speed_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #2563eb;")

        self._btn_speed_reset = QPushButton("1.5x")
        self._btn_speed_reset.setFixedHeight(24)
        self._btn_speed_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_speed_reset.setToolTip("Reset to default 1.50x speed")
        self._btn_speed_reset.clicked.connect(lambda: self._slider_speed.setValue(150))

        self._btn_redub = QPushButton(f"⚡ Re-dub Video at {settings.tts_speed:.2f}x")
        self._btn_redub.setObjectName("primaryButton")
        self._btn_redub.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_redub.setToolTip("Re-synthesize Khmer speech at this speed and re-render the dubbed video")
        self._btn_redub.clicked.connect(self._on_redub_speed_clicked)

        self._lbl_redub_status = QLabel("")
        self._lbl_redub_status.setStyleSheet("font-size: 12px; color: #64748b; font-weight: 500;")

        speed_layout.addWidget(lbl_speed)
        speed_layout.addWidget(self._btn_speed_down)
        speed_layout.addWidget(self._slider_speed)
        speed_layout.addWidget(self._btn_speed_up)
        speed_layout.addWidget(self._lbl_speed_val)
        speed_layout.addWidget(self._btn_speed_reset)
        speed_layout.addSpacing(6)
        speed_layout.addWidget(self._btn_redub)
        speed_layout.addWidget(self._lbl_redub_status)
        speed_layout.addStretch(1)

        main_layout.addWidget(speed_panel)

        # 4. Bottom Action Bar
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.setSpacing(10)

        self._btn_open_folder = QPushButton("Open File Location")
        self._btn_open_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_open_folder.clicked.connect(self._on_open_folder_clicked)
        footer_layout.addWidget(self._btn_open_folder)

        self._btn_play = QPushButton("▶ Play Dubbed Video")
        self._btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_play.clicked.connect(self._on_play_video_clicked)
        has_video = bool(self.episode.output_video_path and Path(self.episode.output_video_path).exists())
        self._btn_play.setEnabled(has_video)
        footer_layout.addWidget(self._btn_play)

        footer_layout.addStretch(1)

        btn_close = QPushButton("Close")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self._on_close_clicked)
        footer_layout.addWidget(btn_close)

        main_layout.addLayout(footer_layout)

    def _build_speech_editor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Editor Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        # Search Bar
        self._txt_search = QLineEdit()
        self._txt_search.setPlaceholderText("Search dialogue or Khmer translation...")
        self._txt_search.setClearButtonEnabled(True)
        self._txt_search.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self._txt_search, stretch=1)

        # Segment Count Badge
        self._lbl_segment_count = QLabel("0 Speech Lines")
        self._lbl_segment_count.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #64748b; padding: 4px 8px; "
            "background-color: #f1f5f9; border-radius: 4px;"
        )
        toolbar.addWidget(self._lbl_segment_count)

        # CRUD Action Buttons
        self._btn_add = QPushButton("+ Add Speech")
        self._btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_add.clicked.connect(self._on_add_line_clicked)
        toolbar.addWidget(self._btn_add)

        self._btn_edit = QPushButton("Edit Selected")
        self._btn_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_edit.setEnabled(False)
        self._btn_edit.clicked.connect(self._on_edit_line_clicked)
        toolbar.addWidget(self._btn_edit)

        self._btn_delete = QPushButton("Delete Selected")
        self._btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_delete.setStyleSheet("color: #dc2626;")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._on_delete_line_clicked)
        toolbar.addWidget(self._btn_delete)

        self._btn_save_changes = QPushButton("Save Changes")
        self._btn_save_changes.setObjectName("primaryButton")
        self._btn_save_changes.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_save_changes.setEnabled(False)
        self._btn_save_changes.clicked.connect(self._on_save_translations_clicked)
        toolbar.addWidget(self._btn_save_changes)

        layout.addLayout(toolbar)

        # Speech Table
        self._tbl_speech = QTableWidget()
        self._tbl_speech.setColumnCount(5)
        self._tbl_speech.setHorizontalHeaderLabels(
            ["#", "Start", "End", "Original Dialogue", "Khmer Translation"]
        )
        self._tbl_speech.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._tbl_speech.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._tbl_speech.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._tbl_speech.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._tbl_speech.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._tbl_speech.setColumnWidth(0, 48)
        self._tbl_speech.setColumnWidth(1, 85)
        self._tbl_speech.setColumnWidth(2, 85)
        self._tbl_speech.verticalHeader().setVisible(False)
        self._tbl_speech.verticalHeader().setMinimumSectionSize(38)
        self._tbl_speech.setWordWrap(True)
        self._tbl_speech.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._tbl_speech.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._tbl_speech.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tbl_speech.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._tbl_speech.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self._tbl_speech.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tbl_speech.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tbl_speech.cellDoubleClicked.connect(self._on_table_double_clicked)
        self._tbl_speech.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self._tbl_speech, stretch=1)

        # Editor Status Bar
        editor_footer = QHBoxLayout()
        self._lbl_editor_status = QLabel("Ready.")
        self._lbl_editor_status.setStyleSheet("font-size: 12px; color: #64748b;")
        editor_footer.addWidget(self._lbl_editor_status)
        editor_footer.addStretch(1)

        self._lbl_unsaved_warning = QLabel("")
        self._lbl_unsaved_warning.setStyleSheet("font-size: 12px; font-weight: 600; color: #d97706;")
        editor_footer.addWidget(self._lbl_unsaved_warning)

        layout.addLayout(editor_footer)

        return widget

    def _build_preview_tab(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.verticalScrollBar().setSingleStep(16)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Metadata Card
        meta_widget = QWidget()
        meta_widget.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px;"
        )
        meta_layout = QVBoxLayout(meta_widget)
        meta_layout.setSpacing(8)

        source_name = Path(self.episode.source_file).name if self.episode.source_file else "Unknown"
        self._lbl_source = QLabel(f"<b>Source:</b> {source_name}")
        self._lbl_source.setWordWrap(True)
        self._lbl_source.setStyleSheet("font-size: 13px; color: #334155;")

        lang_str = self.episode.detected_language or "Not probed yet"
        conf_str = (
            f" ({int(self.episode.language_confidence * 100)}% confidence)"
            if self.episode.language_confidence
            else ""
        )
        self._lbl_language = QLabel(f"<b>Spoken Language:</b> {lang_str.upper()}{conf_str}")
        self._lbl_language.setWordWrap(True)
        self._lbl_language.setStyleSheet("font-size: 13px; color: #334155;")

        meta_layout.addWidget(self._lbl_source)
        meta_layout.addWidget(self._lbl_language)

        mute_status = (
            f"Muted during dialogue (Background sound: {int(settings.original_audio_volume * 100)}%)"
            if settings.mute_original_speaker
            else f"Blended ({int(settings.original_audio_volume * 100)}% background)"
        )
        lbl_audio_mode = QLabel(
            f"<b>Original Speaker:</b> <span style='color: #2563eb; font-weight: 600;'>{mute_status}</span>"
        )
        lbl_audio_mode.setWordWrap(True)
        lbl_audio_mode.setStyleSheet("font-size: 13px; color: #334155;")
        meta_layout.addWidget(lbl_audio_mode)

        lbl_tts_info = QLabel(
            f"<b>TTS Voice Settings:</b> Volume {int(settings.tts_volume * 100)}% | Speed {settings.tts_speed:.2f}x | Background Sound: {int(settings.original_audio_volume * 100)}%"
        )
        lbl_tts_info.setWordWrap(True)
        lbl_tts_info.setStyleSheet("font-size: 13px; color: #334155;")
        meta_layout.addWidget(lbl_tts_info)

        if self.episode.output_video_path and Path(self.episode.output_video_path).exists():
            out_vid_name = Path(self.episode.output_video_path).name
            lbl_out = QLabel(f"<b>Dubbed Video:</b> <span style='color: #16a34a;'>{out_vid_name}</span>")
            lbl_out.setWordWrap(True)
            lbl_out.setStyleSheet("font-size: 13px; color: #334155;")
            meta_layout.addWidget(lbl_out)

        layout.addWidget(meta_widget)

        # Raw Subtitles Preview
        preview_header = QLabel("Generated Subtitles (.srt / .ass) File Preview:")
        preview_header.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        layout.addWidget(preview_header)

        self._txt_preview = QPlainTextEdit()
        self._txt_preview.setReadOnly(True)
        self._txt_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._txt_preview.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._txt_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._txt_preview.setMinimumHeight(240)
        self._txt_preview.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #cbd5e1; font-family: monospace; font-size: 12px;"
        )
        layout.addWidget(self._txt_preview, stretch=1)

        scroll_area.setWidget(widget)
        return scroll_area

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    def _resolve_translation_file(self) -> Path | None:
        """Find the existing or expected translation JSON path for this episode."""
        if self.episode.translation_path and Path(self.episode.translation_path).exists():
            return Path(self.episode.translation_path)

        # Fallback to standard cache directory
        cache_file = settings.cache_dir / f"episode_{self.episode.id}" / f"translation_{self.episode.id}.json"
        if cache_file.exists():
            return cache_file
        return None

    def _load_segments(self) -> None:
        """Load speech segments from translation JSON or fallback to transcript."""
        target_json = self._resolve_translation_file()
        self._segments = []

        if target_json and target_json.exists():
            try:
                with target_json.open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for idx, item in enumerate(raw_data, start=1):
                        if isinstance(item, dict):
                            item_id = item.get("id", idx)
                            start = float(item.get("start", 0.0))
                            end = float(item.get("end", start + 1.0))
                            text = str(item.get("text") or item.get("original_text", "")).strip()
                            translated = str(item.get("translated_text", "")).strip()
                            self._segments.append(
                                {
                                    "id": item_id,
                                    "start": start,
                                    "end": end,
                                    "text": text,
                                    "translated_text": translated,
                                    "speaker": item.get("speaker"),
                                }
                            )
            except Exception as exc:
                logger.error("Failed to load translation segments: {}", exc)
                self._lbl_editor_status.setText(f"Error loading translation: {exc}")
        elif self.episode.transcript_path and Path(self.episode.transcript_path).exists():
            # Load from transcript if translated not generated yet
            try:
                with Path(self.episode.transcript_path).open("r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for idx, item in enumerate(raw_data, start=1):
                        if isinstance(item, dict):
                            self._segments.append(
                                {
                                    "id": item.get("id", idx),
                                    "start": float(item.get("start", 0.0)),
                                    "end": float(item.get("end", 1.0)),
                                    "text": str(item.get("text", "")).strip(),
                                    "translated_text": "",
                                }
                            )
                    self._lbl_editor_status.setText(
                        "Loaded transcript segments. (Translation not run yet — Khmer text is empty)."
                    )
            except Exception as exc:
                logger.error("Failed to load transcript: {}", exc)

        self._populate_speech_table()

    def _populate_speech_table(self) -> None:
        """Populate the speech table filtering by the search box."""
        query = self._txt_search.text().strip().lower()

        self._filtered_indices = []
        for i, seg in enumerate(self._segments):
            if not query:
                self._filtered_indices.append(i)
            else:
                orig = seg.get("text", "").lower()
                trans = seg.get("translated_text", "").lower()
                if query in orig or query in trans:
                    self._filtered_indices.append(i)

        self._tbl_speech.setRowCount(len(self._filtered_indices))

        for row, idx in enumerate(self._filtered_indices):
            seg = self._segments[idx]

            # Col 0: ID
            item_id = QTableWidgetItem(str(seg.get("id", row + 1)))
            item_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._tbl_speech.setItem(row, 0, item_id)

            # Col 1: Start Time
            start_str = format_timestamp(seg.get("start", 0.0))
            item_start = QTableWidgetItem(start_str)
            item_start.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._tbl_speech.setItem(row, 1, item_start)

            # Col 2: End Time
            end_str = format_timestamp(seg.get("end", 0.0))
            item_end = QTableWidgetItem(end_str)
            item_end.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._tbl_speech.setItem(row, 2, item_end)

            # Col 3: Original Text
            orig_text = seg.get("text", "")
            item_orig = QTableWidgetItem(orig_text)
            item_orig.setToolTip(orig_text)
            item_orig.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._tbl_speech.setItem(row, 3, item_orig)

            # Col 4: Translated Khmer Speech
            trans_text = seg.get("translated_text", "")
            item_trans = QTableWidgetItem(trans_text)
            item_trans.setToolTip(trans_text)
            item_trans.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self._tbl_speech.setItem(row, 4, item_trans)

        self._tbl_speech.resizeRowsToContents()

        total_lines = len(self._segments)
        filter_text = f" (filtered {len(self._filtered_indices)})" if query else ""
        self._lbl_segment_count.setText(f"{total_lines} Speech Lines{filter_text}")

        # Update action buttons state
        self._on_table_selection_changed()

    def _on_search_changed(self) -> None:
        self._populate_speech_table()

    def _on_table_selection_changed(self) -> None:
        selected_rows = self._tbl_speech.selectionModel().selectedRows()
        has_sel = len(selected_rows) > 0
        self._btn_edit.setEnabled(has_sel)
        self._btn_delete.setEnabled(has_sel)

    def _on_table_double_clicked(self, row: int, _col: int) -> None:
        self._edit_row(row)

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------
    def _on_add_line_clicked(self) -> None:
        """Create a new speech line."""
        last_end = 0.0
        if self._segments:
            last_end = max(seg.get("end", 0.0) for seg in self._segments)

        default_start = round(last_end + 0.2, 3)
        default_end = round(default_start + 2.0, 3)

        dlg = SpeechSegmentEditDialog(
            segment=None,
            default_start=default_start,
            default_end=default_end,
            parent=self,
        )
        if dlg.exec():
            new_data = dlg.get_segment_data()
            new_data["id"] = len(self._segments) + 1
            self._segments.append(new_data)
            # Sort by start timestamp
            self._segments.sort(key=lambda s: s.get("start", 0.0))
            # Re-index
            for i, s in enumerate(self._segments, start=1):
                s["id"] = i

            self._set_unsaved(True)
            self._populate_speech_table()
            self._lbl_editor_status.setText(f"Added new speech line at {format_timestamp(new_data['start'])}.")

    def _on_edit_line_clicked(self) -> None:
        """Edit the currently selected speech line."""
        selected_rows = self._tbl_speech.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        self._edit_row(row)

    def _edit_row(self, row: int) -> None:
        if not (0 <= row < len(self._filtered_indices)):
            return
        actual_idx = self._filtered_indices[row]
        segment = self._segments[actual_idx]

        dlg = SpeechSegmentEditDialog(
            segment=segment,
            parent=self,
        )
        if dlg.exec():
            updated = dlg.get_segment_data()
            self._segments[actual_idx] = updated
            # Re-sort
            self._segments.sort(key=lambda s: s.get("start", 0.0))
            for i, s in enumerate(self._segments, start=1):
                s["id"] = i

            self._set_unsaved(True)
            self._populate_speech_table()
            self._lbl_editor_status.setText(f"Updated speech line #{updated.get('id', row + 1)}.")

    def _on_delete_line_clicked(self) -> None:
        """Delete the selected speech line(s)."""
        selected_rows = self._tbl_speech.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        if not (0 <= row < len(self._filtered_indices)):
            return

        actual_idx = self._filtered_indices[row]
        seg = self._segments[actual_idx]

        reply = QMessageBox.question(
            self,
            "Delete Speech Line",
            f"Are you sure you want to delete speech line #{seg.get('id', row + 1)}?\n\n"
            f"\"{seg.get('translated_text') or seg.get('text', '')}\"",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._segments.pop(actual_idx)
            for i, s in enumerate(self._segments, start=1):
                s["id"] = i

            self._set_unsaved(True)
            self._populate_speech_table()
            self._lbl_editor_status.setText("Deleted speech line.")

    def _set_unsaved(self, unsaved: bool) -> None:
        self._has_unsaved_changes = unsaved
        self._btn_save_changes.setEnabled(unsaved)
        if unsaved:
            self._lbl_unsaved_warning.setText("● Unsaved Changes")
        else:
            self._lbl_unsaved_warning.setText("")

    # -------------------------------------------------------------------------
    # Saving & Subtitle Synchronization
    # -------------------------------------------------------------------------
    def _on_save_translations_clicked(self) -> None:
        """Persist translated speech back to JSON and synchronize subtitles."""
        target_path = self._resolve_translation_file()
        if not target_path:
            work_dir = settings.cache_dir / f"episode_{self.episode.id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            target_path = work_dir / f"translation_{self.episode.id}.json"
            self.episode.translation_path = str(target_path)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(self._segments, f, ensure_ascii=False, indent=2)

            # Synchronize subtitle files (.srt and .ass) if paths exist or configured
            sub_msg = self._sync_subtitles_on_disk()

            self._set_unsaved(False)
            self._lbl_editor_status.setText(
                f"Saved {len(self._segments)} speech lines successfully. {sub_msg}"
            )
            # Refresh preview tab text
            self._load_episode_data()
        except Exception as exc:
            logger.error("Failed to save translation changes: {}", exc)
            self._lbl_editor_status.setText(f"Failed to save changes: {exc}")
            QMessageBox.critical(self, "Save Error", f"Unable to save changes: {exc}")

    def _sync_subtitles_on_disk(self) -> str:
        """Update .srt and .ass files to reflect the edited speech translations."""
        if not self._segments:
            return ""

        updated_files: list[str] = []

        # 1. Update SRT if exists or path assigned
        srt_path = self.episode.subtitle_srt_path
        if not srt_path and self.episode.output_video_path:
            srt_path = str(Path(self.episode.output_video_path).with_suffix(".srt"))

        if srt_path:
            try:
                p = Path(srt_path)
                create_srt_file(self._segments, p)
                self.episode.subtitle_srt_path = str(p)
                updated_files.append(".srt")
            except Exception as exc:
                logger.warning("Failed updating SRT after speech edit: {}", exc)

        # 2. Update ASS if exists
        ass_path = self.episode.subtitle_ass_path
        if not ass_path and self.episode.output_video_path:
            ass_path = str(Path(self.episode.output_video_path).with_suffix(".ass"))

        if ass_path:
            try:
                p = Path(ass_path)
                create_ass_file(self._segments, p)
                self.episode.subtitle_ass_path = str(p)
                updated_files.append(".ass")
            except Exception as exc:
                logger.warning("Failed updating ASS after speech edit: {}", exc)

        return f"Subtitles synchronized ({', '.join(updated_files)})." if updated_files else ""

    # -------------------------------------------------------------------------
    # Preview & Media Actions
    # -------------------------------------------------------------------------
    def _load_episode_data(self) -> None:
        """Load compiled subtitle content into the preview text box."""
        srt_path = self.episode.subtitle_srt_path
        ass_path = self.episode.subtitle_ass_path

        target_file = None
        if srt_path and Path(srt_path).exists():
            target_file = Path(srt_path)
        elif ass_path and Path(ass_path).exists():
            target_file = Path(ass_path)

        if target_file and target_file.exists():
            try:
                content = target_file.read_text(encoding="utf-8")
                self._txt_preview.setPlainText(content[:15000])
            except Exception as exc:
                self._txt_preview.setPlainText(f"Failed to read subtitle file: {exc}")
        elif self._segments:
            lines = []
            for s in self._segments[:500]:
                start_sec = s.get("start", 0.0)
                m, sec = divmod(int(start_sec), 60)
                text = s.get("translated_text", "")
                lines.append(f"[{m:02d}:{sec:02d}] {text}")
            self._txt_preview.setPlainText("\n".join(lines))
        else:
            if self.episode.status == EpisodeStatus.DONE:
                self._txt_preview.setPlainText("Processing completed successfully.")
            else:
                self._txt_preview.setPlainText(
                    "Dubbed dialogue and subtitles have not been generated for this episode yet."
                )

    def _on_play_video_clicked(self) -> None:
        if self.episode.output_video_path:
            p = Path(self.episode.output_video_path).resolve()
            if p.exists() and p.is_file():
                try:
                    if sys.platform == "win32":
                        os.startfile(str(p))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(p)])
                    else:
                        subprocess.Popen(["xdg-open", str(p)])
                except Exception as exc:
                    logger.error("Failed to open media player for {}: {}", p, exc)

    def _on_speed_slider_changed(self, value: int) -> None:
        speed = round(value / 100.0, 2)
        settings.tts_speed = speed
        self._lbl_speed_val.setText(f"{speed:.2f}x")
        self._btn_redub.setText(f"⚡ Re-dub Video at {speed:.2f}x")

    def _on_speed_decrease_clicked(self) -> None:
        new_val = max(50, self._slider_speed.value() - 5)
        self._slider_speed.setValue(new_val)

    def _on_speed_increase_clicked(self) -> None:
        new_val = min(200, self._slider_speed.value() + 5)
        self._slider_speed.setValue(new_val)

    def _on_redub_speed_clicked(self) -> None:
        """Trigger rapid re-dubbing with the chosen speech speed."""
        if self._has_unsaved_changes:
            self._on_save_translations_clicked()

        speed = round(self._slider_speed.value() / 100.0, 2)
        settings.tts_speed = speed

        if self._job_manager:
            self._btn_redub.setEnabled(False)
            self._btn_redub.setText("⏳ Re-dubbing...")
            self._lbl_redub_status.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: 500;")
            self._lbl_redub_status.setText(f"Starting re-dub at {speed:.2f}x...")
            self._job_manager.redub_episode(self.episode.id, speed)
        else:
            QMessageBox.information(
                self,
                "Speed Applied",
                f"Speech speed set to {speed:.2f}x. You can run the episode from the workspace.",
            )

    def _on_job_progress(self, episode_id: int, stage_desc: str, pct: int) -> None:
        if episode_id == self.episode.id:
            self._lbl_redub_status.setStyleSheet("font-size: 11px; color: #2563eb; font-weight: 500;")
            self._lbl_redub_status.setText(f"{stage_desc} ({pct}%)")

    def _on_job_completed(self, episode_id: int) -> None:
        if episode_id == self.episode.id:
            speed = round(self._slider_speed.value() / 100.0, 2)
            self._btn_redub.setEnabled(True)
            self._btn_redub.setText(f"⚡ Re-dub Video at {speed:.2f}x")
            self._lbl_redub_status.setStyleSheet("font-size: 11px; color: #16a34a; font-weight: 600;")
            self._lbl_redub_status.setText(f"✓ Re-dubbed at {speed:.2f}x successfully!")
            self._btn_play.setEnabled(True)
            self._status_badge.setText("Done")
            self._status_badge.setStyleSheet(
                "font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; "
                "border: 1px solid #16a34a; color: #16a34a;"
            )
            # Re-fetch episode data to get updated output_video_path
            self._load_episode_data()

    def _on_job_failed(self, episode_id: int, error_message: str) -> None:
        if episode_id == self.episode.id:
            speed = round(self._slider_speed.value() / 100.0, 2)
            self._btn_redub.setEnabled(True)
            self._btn_redub.setText(f"⚡ Re-dub Video at {speed:.2f}x")
            self._lbl_redub_status.setStyleSheet("font-size: 11px; color: #dc2626; font-weight: 500;")
            self._lbl_redub_status.setText(f"Failed: {error_message[:40]}")

    def _on_open_folder_clicked(self) -> None:
        target_path_str = (
            self.episode.output_video_path
            or self.episode.subtitle_srt_path
            or self.episode.source_file
        )
        if not target_path_str:
            return

        target_path = Path(target_path_str)
        if not target_path.exists():
            return

        parent_dir = target_path.parent.resolve()
        if not parent_dir.is_dir():
            logger.warning("Target parent path is not a directory: {}", parent_dir)
            return

        allowed_roots = [
            settings.output_root.resolve(),
            settings.cache_dir.resolve(),
        ]
        if self.episode.source_file:
            try:
                allowed_roots.append(Path(self.episode.source_file).parent.resolve())
            except Exception:
                pass

        is_allowed = any(
            parent_dir == root or parent_dir.is_relative_to(root)
            for root in allowed_roots
        )
        if not is_allowed:
            logger.warning("Folder path '{}' is outside allowed directories; refusing to open", parent_dir)
            return

        try:
            if sys.platform == "win32":
                os.startfile(str(parent_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(parent_dir)])
            else:
                subprocess.Popen(["xdg-open", str(parent_dir)])
        except Exception as exc:
            logger.error("Failed to open file explorer for directory '{}': {}", parent_dir, exc)

    def _on_close_clicked(self) -> None:
        if self._has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes to translated speech. Do you want to save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_translations_clicked()
                self.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                self.reject()
            return
        self.accept()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self._has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes to translated speech. Do you want to save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_translations_clicked()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
                return
        event.accept()
