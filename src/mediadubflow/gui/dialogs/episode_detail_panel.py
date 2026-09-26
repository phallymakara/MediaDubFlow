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

from mediadubflow.config.settings import settings
from mediadubflow.models.orm import Episode, EpisodeStatus


class EpisodeDetailPanel(QDialog):
    """Inspector dialog for reviewing episode checkpoints and subtitle output."""

    def __init__(self, episode: Episode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.episode = episode
        self.setWindowTitle(f"Episode {episode.episode_number:02d} Review — MediaDubFlow")
        self.setMinimumSize(640, 420)
        self.resize(720, 520)
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

        if self.episode.output_video_path and Path(self.episode.output_video_path).exists():
            out_vid_name = Path(self.episode.output_video_path).name
            lbl_out = QLabel(f"<b>Dubbed Video:</b> <span style='color: #16a34a;'>{out_vid_name}</span>")
            lbl_out.setStyleSheet("font-size: 13px; color: #334155;")
            meta_layout.addWidget(lbl_out)

        main_layout.addWidget(meta_widget)

        # Content Preview Section
        preview_header = QLabel("Generated Khmer Dialogue / Subtitles Preview:")
        preview_header.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        main_layout.addWidget(preview_header)

        self._txt_preview = QPlainTextEdit()
        self._txt_preview.setReadOnly(True)
        self._txt_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._txt_preview.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._txt_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._txt_preview.verticalScrollBar().setSingleStep(16)
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

        if self.episode.output_video_path and Path(self.episode.output_video_path).exists():
            self._btn_play = QPushButton("▶ Play Dubbed Video")
            self._btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
            self._btn_play.clicked.connect(self._on_play_video_clicked)
            btn_layout.addWidget(self._btn_play)

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
        elif self.episode.translation_path and Path(self.episode.translation_path).exists():
            try:
                import json  # noqa: PLC0415
                with open(self.episode.translation_path, "r", encoding="utf-8") as f:
                    segments = json.load(f)
                lines = []
                for s in segments:
                    start_sec = s.get("start", 0.0)
                    m, sec = divmod(int(start_sec), 60)
                    text = s.get("translated_text", "")
                    lines.append(f"[{m:02d}:{sec:02d}] {text}")
                self._txt_preview.setPlainText("\n".join(lines[:500]))
            except Exception as exc:
                self._txt_preview.setPlainText(f"Could not load dialogue: {exc}")
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

    def _on_open_folder_clicked(self) -> None:
        target_path_str = self.episode.output_video_path or self.episode.subtitle_srt_path or self.episode.source_file
        if not target_path_str:
            return

        target_path = Path(target_path_str)
        if not target_path.exists():
            return

        parent_dir = target_path.parent.resolve()
        if not parent_dir.is_dir():
            logger.warning("Target parent path is not a directory: {}", parent_dir)
            return

        # Ensure parent_dir is contained within allowed workspace/media roots
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
