"""
Episode Detail Panel — Detailed inspector for episode status and generated subtitles.

Provides:
- Comprehensive episode metadata (detected language, audio format, status).
- Pipeline checkpoint paths (extracted audio, transcript, translation, subtitle).
- Live preview of generated Khmer subtitles (.srt / .ass).
- Quick action to open output files in the system file explorer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.models.orm import Episode, EpisodeStatus


class EpisodeDetailPanel(QDialog):
    """Inspector dialog for reviewing episode checkpoints and subtitle output."""

    def __init__(self, episode: Episode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.episode = episode
        self.setWindowTitle(f"Episode {episode.episode_number:02d} Review — MediaDubFlow")
        self.setMinimumSize(700, 520)
        self.setModal(True)

        self._setup_ui()
        self._load_episode_data()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel(f"Episode {self.episode.episode_number:02d} Inspector")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #0f172a;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        self._status_badge = QLabel(self.episode.status.value.replace("_", " ").title())
        self._status_badge.setStyleSheet(
            "font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; "
            "border: 1px solid #2563eb; color: #2563eb;"
        )
        header_layout.addWidget(self._status_badge)
        main_layout.addLayout(header_layout)

        # Metadata Card
        meta_widget = QWidget()
        meta_widget.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px;"
        )
        meta_layout = QVBoxLayout(meta_widget)
        meta_layout.setSpacing(6)

        source_name = Path(self.episode.source_file).name if self.episode.source_file else "Unknown"
        self._lbl_source = QLabel(f"<b>Source:</b> {source_name}")
        self._lbl_source.setStyleSheet("font-size: 13px; color: #334155;")

        lang_str = self.episode.detected_language or "Not probed yet"
        conf_str = (
            f" ({int(self.episode.language_confidence * 100)}% confidence)"
            if self.episode.language_confidence
            else ""
        )
        self._lbl_language = QLabel(f"<b>Spoken Language:</b> {lang_str.upper()}{conf_str}")
        self._lbl_language.setStyleSheet("font-size: 13px; color: #334155;")

        meta_layout.addWidget(self._lbl_source)
        meta_layout.addWidget(self._lbl_language)
        main_layout.addWidget(meta_widget)

        # Subtitle Preview Section
        preview_header = QLabel("Generated Khmer Subtitles Preview (.srt / .ass):")
        preview_header.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        main_layout.addWidget(preview_header)

        self._txt_preview = QPlainTextEdit()
        self._txt_preview.setReadOnly(True)
        self._txt_preview.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #cbd5e1; font-family: monospace; font-size: 12px;"
        )
        main_layout.addWidget(self._txt_preview, stretch=1)

        # Action Buttons
        btn_layout = QHBoxLayout()

        self._btn_open_folder = QPushButton("Open File Location")
        self._btn_open_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_open_folder.clicked.connect(self._on_open_folder_clicked)
        btn_layout.addWidget(self._btn_open_folder)

        btn_layout.addStretch(1)

        btn_close = QPushButton("Close")
        btn_close.setObjectName("primaryButton")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        main_layout.addLayout(btn_layout)

    def _load_episode_data(self) -> None:
        # Load subtitle content if available
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
                self._txt_preview.setPlainText(content[:10000])  # limit preview
            except Exception as exc:
                self._txt_preview.setPlainText(f"Failed to read subtitle file: {exc}")
        else:
            if self.episode.status == EpisodeStatus.DONE:
                self._txt_preview.setPlainText("Subtitles completed but file path not found.")
            else:
                self._txt_preview.setPlainText(
                    "Subtitles have not been generated for this episode yet."
                )

    def _on_open_folder_clicked(self) -> None:
        target_path_str = self.episode.subtitle_srt_path or self.episode.source_file
        if not target_path_str:
            return

        target_path = Path(target_path_str)
        if not target_path.exists():
            return

        parent_dir = target_path.parent.resolve()
        if not parent_dir.is_dir():
            logger.warning("Target parent path is not a directory: {}", parent_dir)
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
