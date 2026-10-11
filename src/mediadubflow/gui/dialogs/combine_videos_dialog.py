"""
Combine Videos Dialog — Merge multiple dubbed or source episodes into multi-part or single full video files.

Supports:
- Combining episodes in batches (e.g. 3 videos into 1, or custom N-in-1).
- Combining all episodes in the project into a single full video file.
- Choosing between final Khmer dubbed videos and original source videos.
- Non-blocking asynchronous FFmpeg execution with live progress updates.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config.settings import settings
from mediadubflow.models.orm import Episode, Project
from mediadubflow.utils.ffmpeg import concatenate_videos, kill_all_ffmpeg_processes

if TYPE_CHECKING:
    from mediadubflow.gui.async_bridge import AsyncBridge


class CombineBatchItem:
    """Represents a single planned combined video file."""

    def __init__(
        self,
        episodes: list[Episode],
        input_paths: list[Path],
        output_filename: str,
        start_ep: int,
        end_ep: int,
    ) -> None:
        self.episodes = episodes
        self.input_paths = input_paths
        self.output_filename = output_filename
        self.start_ep = start_ep
        self.end_ep = end_ep


class CombineVideosDialog(QDialog):
    """Modal dialog allowing users to combine multiple episodes into single files."""

    def __init__(
        self,
        project: Project,
        episodes: list[Episode],
        bridge: AsyncBridge,
        initial_mode: str = "batch",  # "batch" or "all"
        initial_batch_size: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Combine Videos")
        self.resize(680, 560)

        self.setMinimumSize(580, 480)

        self._project = project
        self._episodes = sorted(
            episodes,
            key=lambda e: (e.episode_number if e.episode_number > 0 else 999999, e.id),
        )
        self._bridge = bridge
        self._initial_mode = initial_mode
        self._is_combining = False
        self._cancelled = False

        self._batches: list[CombineBatchItem] = []

        self._setup_ui(initial_batch_size)
        self._refresh_plan()

    def _setup_ui(self, initial_batch_size: int) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # 1. Header Information
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel(f"Series: {self._project.name}")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0f172a;")
        lbl_subtitle = QLabel(
            f"Total Episodes: {len(self._episodes)}  |  Target Language: Khmer (km)"
        )
        lbl_subtitle.setStyleSheet("font-size: 12px; color: #64748b;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)
        main_layout.addLayout(title_box)

        # 1b. Title of Movie Configuration
        title_config_row = QHBoxLayout()
        title_config_row.setSpacing(8)
        lbl_movie = QLabel("Movie Title:")
        lbl_movie.setStyleSheet("font-size: 12px; font-weight: 600; color: #334155;")
        self._txt_movie_title = QLineEdit(self._resolve_series_title())
        self._txt_movie_title.setToolTip("Movie title placed after [ភាគ --] in the output filename")
        self._txt_movie_title.textChanged.connect(self._refresh_plan)
        title_config_row.addWidget(lbl_movie)
        title_config_row.addWidget(self._txt_movie_title, stretch=1)
        main_layout.addLayout(title_config_row)

        # 2. Source Video Selection
        source_frame = QFrame()
        source_frame.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px;"
        )
        source_layout = QVBoxLayout(source_frame)
        source_layout.setContentsMargins(10, 8, 10, 8)
        source_layout.setSpacing(6)

        lbl_source_head = QLabel("Select Video Source to Combine:")
        lbl_source_head.setStyleSheet("font-size: 12px; font-weight: 600; color: #334155;")
        source_layout.addWidget(lbl_source_head)

        dubbed_count = sum(
            1
            for ep in self._episodes
            if ep.output_video_path
            and Path(ep.output_video_path).is_file()
            and Path(ep.output_video_path).stat().st_size > 0
        )
        source_count = sum(
            1
            for ep in self._episodes
            if ep.source_file
            and Path(ep.source_file).is_file()
            and Path(ep.source_file).stat().st_size > 0
        )

        self._rb_use_dubbed = QRadioButton(
            f"Use Khmer Dubbed Videos ({dubbed_count}/{len(self._episodes)} ready)"
        )
        self._rb_use_source = QRadioButton(
            f"Use Original Source Videos ({source_count}/{len(self._episodes)} available)"
        )

        self._source_group = QButtonGroup(self)
        self._source_group.addButton(self._rb_use_dubbed)
        self._source_group.addButton(self._rb_use_source)

        if dubbed_count > 0:
            self._rb_use_dubbed.setChecked(True)
        else:
            self._rb_use_source.setChecked(True)
            self._rb_use_dubbed.setEnabled(False)

        self._rb_use_dubbed.toggled.connect(self._refresh_plan)
        self._rb_use_source.toggled.connect(self._refresh_plan)

        source_layout.addWidget(self._rb_use_dubbed)
        source_layout.addWidget(self._rb_use_source)
        main_layout.addWidget(source_frame)

        # 3. Combination Mode, Batch Size & Starting ភាគ Number
        mode_frame = QFrame()
        mode_frame.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px;"
        )
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setContentsMargins(10, 8, 10, 8)
        mode_layout.setSpacing(8)

        lbl_mode_head = QLabel("Combination Mode & ភាគ Numbering:")
        lbl_mode_head.setStyleSheet("font-size: 12px; font-weight: 600; color: #334155;")
        mode_layout.addWidget(lbl_mode_head)

        # Mode A: Batch (N in 1)
        batch_row = QHBoxLayout()
        batch_row.setSpacing(8)
        self._rb_mode_batch = QRadioButton("Combine in groups:")
        self._rb_mode_batch.setChecked(self._initial_mode == "batch")
        self._rb_mode_batch.toggled.connect(self._on_mode_toggled)

        self._spin_batch_size = QSpinBox()
        self._spin_batch_size.setRange(2, 500)
        self._spin_batch_size.setValue(max(2, initial_batch_size))
        self._spin_batch_size.setSuffix(" videos into 1")
        self._spin_batch_size.setFixedWidth(150)
        self._spin_batch_size.valueChanged.connect(self._refresh_plan)

        batch_row.addWidget(self._rb_mode_batch)
        batch_row.addWidget(self._spin_batch_size)

        batch_row.addSpacing(16)
        lbl_start = QLabel("Start ភាគ at:")
        lbl_start.setStyleSheet("font-size: 12px; font-weight: 600; color: #475569;")
        self._spin_start_part = QSpinBox()
        self._spin_start_part.setRange(1, 9999)
        self._spin_start_part.setValue(1)
        self._spin_start_part.setFixedWidth(70)
        self._spin_start_part.setToolTip("Starting ភាគ number for the combined video sequence (e.g. 1 for [ភាគ 01])")
        self._spin_start_part.valueChanged.connect(self._refresh_plan)

        batch_row.addWidget(lbl_start)
        batch_row.addWidget(self._spin_start_part)
        batch_row.addStretch(1)
        mode_layout.addLayout(batch_row)

        # Mode B: Combine ALL into 1
        self._rb_mode_all = QRadioButton("Combine ALL episodes into 1 video file")
        self._rb_mode_all.setChecked(self._initial_mode == "all")
        self._rb_mode_all.toggled.connect(self._on_mode_toggled)
        mode_layout.addWidget(self._rb_mode_all)

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._rb_mode_batch)
        self._mode_group.addButton(self._rb_mode_all)

        main_layout.addWidget(mode_frame)


        # 4. Destination Directory Row
        dest_row = QHBoxLayout()
        dest_row.setSpacing(8)
        lbl_dest = QLabel("Destination:")
        lbl_dest.setStyleSheet("font-size: 12px; font-weight: 600; color: #475569;")

        dest_folder = (
            Path(self._project.output_folder).resolve()
            if self._project.output_folder
            else settings.output_root
        )
        self._txt_dest = QLineEdit(str(dest_folder))
        self._txt_dest.textChanged.connect(self._refresh_plan)

        self._btn_browse = QPushButton("Browse...")
        self._btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_browse.clicked.connect(self._on_browse_dest_clicked)

        dest_row.addWidget(lbl_dest)
        dest_row.addWidget(self._txt_dest, stretch=1)
        dest_row.addWidget(self._btn_browse)
        main_layout.addLayout(dest_row)

        # 5. Planned Outputs Table & Preview
        self._lbl_plan_summary = QLabel("Planned Output Files:")
        self._lbl_plan_summary.setStyleSheet("font-size: 12px; font-weight: 600; color: #334155;")
        main_layout.addWidget(self._lbl_plan_summary)

        self._table_preview = QTableWidget()
        self._table_preview.setColumnCount(3)
        self._table_preview.setHorizontalHeaderLabels(["#", "Output File", "Included Episodes"])
        self._table_preview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table_preview.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table_preview.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table_preview.setColumnWidth(0, 40)
        self._table_preview.verticalHeader().setVisible(False)
        self._table_preview.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_preview.setFixedHeight(140)
        main_layout.addWidget(self._table_preview)

        # 6. Status & Progress Bar
        self._lbl_status = QLabel("Ready to combine.")
        self._lbl_status.setStyleSheet("font-size: 12px; color: #475569;")
        main_layout.addWidget(self._lbl_status)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        main_layout.addWidget(self._progress_bar)

        # 7. Bottom Action Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)

        self._btn_open_folder = QPushButton("Open Output Folder")
        self._btn_open_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_open_folder.setVisible(False)
        self._btn_open_folder.clicked.connect(self._on_open_output_folder_clicked)
        btn_box.addWidget(self._btn_open_folder)

        btn_box.addStretch(1)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)

        self._btn_start = QPushButton("+ Start Combining")
        self._btn_start.setObjectName("primaryButton")
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.clicked.connect(self._on_start_combining_clicked)


        btn_box.addWidget(self._btn_cancel)
        btn_box.addWidget(self._btn_start)
        main_layout.addLayout(btn_box)

    def _on_mode_toggled(self) -> None:
        self._spin_batch_size.setEnabled(self._rb_mode_batch.isChecked())
        self._refresh_plan()

    def _on_browse_dest_clicked(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for Combined Videos",
            self._txt_dest.text(),
        )
        if chosen:
            self._txt_dest.setText(chosen)

    def _resolve_series_title(self) -> str:
        """Resolve localized Khmer series title from cache or project name."""
        project_cache_dir = settings.cache_dir / f"project_{self._project.id}"
        series_title_file = project_cache_dir / "series_title_km.txt"
        raw_title: str | None = None
        if series_title_file.exists():
            try:
                cached = series_title_file.read_text(encoding="utf-8").strip()
                if cached:
                    raw_title = cached
            except OSError:
                pass

        # Fallback to episode Khmer title or project name
        if not raw_title:
            for ep in self._episodes:
                if ep.title and any("\u1780" <= c <= "\u17ff" for c in ep.title):
                    raw_title = ep.title
                    break

        if not raw_title:
            raw_title = self._project.name

        clean_s = re.sub(
            r"^\[\s*(?:ភាគ|ep(?:isode)?|part)\s*[\d០-៩\s\-~_]+\s*(?:ពេញ|all)?\s*\]\s*",
            "",
            raw_title,
            flags=re.IGNORECASE,
        )
        clean_s = re.sub(
            r"[\s_\-]+(?:ភាគ|ep(?:isode)?|part)\s*[\d០-៩\s\-~_]+.*$",
            "",
            clean_s,
            flags=re.IGNORECASE,
        ).strip()
        if clean_s and any("\u1780" <= c <= "\u17ff" for c in clean_s):
            if not clean_s.startswith("រឿង"):
                clean_s = f"រឿង {clean_s}"

        return clean_s if clean_s else self._project.name

    def _refresh_plan(self) -> None:
        """Calculate batch items and update live table preview with [ភាគ --] title format."""
        use_dubbed = self._rb_use_dubbed.isChecked()
        mode_all = self._rb_mode_all.isChecked()
        batch_size = len(self._episodes) if mode_all else self._spin_batch_size.value()

        # Filter valid episodes with available video files
        valid_pairs: list[tuple[Episode, Path]] = []
        for ep in self._episodes:
            candidate: Path | None = None
            if use_dubbed:
                if (
                    ep.output_video_path
                    and Path(ep.output_video_path).is_file()
                    and Path(ep.output_video_path).stat().st_size > 0
                ):
                    candidate = Path(ep.output_video_path).resolve()
            else:
                if (
                    ep.source_file
                    and Path(ep.source_file).is_file()
                    and Path(ep.source_file).stat().st_size > 0
                ):
                    candidate = Path(ep.source_file).resolve()

            if candidate is not None:
                valid_pairs.append((ep, candidate))

        # Get movie title, stripping any leading [ភាគ ...] to prevent duplication
        movie_title = (
            self._txt_movie_title.text().strip()
            if hasattr(self, "_txt_movie_title") and self._txt_movie_title.text().strip()
            else self._resolve_series_title()
        )
        clean_movie_title = re.sub(
            r"^\[\s*(?:ភាគ|ep(?:isode)?|part)\s*[\d០-៩\s\-~_]+\s*(?:ពេញ|all)?\s*\]\s*",
            "",
            movie_title,
            flags=re.IGNORECASE,
        ).strip()
        safe_title = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", clean_movie_title).strip(". ")

        self._batches.clear()
        if not valid_pairs:
            self._table_preview.setRowCount(0)
            self._lbl_plan_summary.setText("No eligible video files found for combination.")
            self._btn_start.setEnabled(False)
            return

        self._btn_start.setEnabled(True)

        start_part = self._spin_start_part.value() if hasattr(self, "_spin_start_part") else 1

        # Build chunks with incrementing [ភាគ --] title of the movie
        for idx, i in enumerate(range(0, len(valid_pairs), batch_size)):
            chunk = valid_pairs[i : i + batch_size]
            chunk_eps = [ep for ep, _ in chunk]
            chunk_paths = [p for _, p in chunk]

            part_num = start_part + idx
            filename = f"[ភាគ {part_num:02d}] {safe_title}.mp4"

            item = CombineBatchItem(
                episodes=chunk_eps,
                input_paths=chunk_paths,
                output_filename=filename,
                start_ep=part_num,
                end_ep=part_num,
            )
            self._batches.append(item)


        # Update preview table
        self._table_preview.setRowCount(len(self._batches))
        for row, batch in enumerate(self._batches):
            it_idx = QTableWidgetItem(str(row + 1))
            it_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table_preview.setItem(row, 0, it_idx)

            it_file = QTableWidgetItem(batch.output_filename)
            self._table_preview.setItem(row, 1, it_file)

            ep_tags = ", ".join(
                f"EP {ep.episode_number:02d}" if ep.episode_number > 0 else f"#{ep.id}"
                for ep in batch.episodes
            )
            it_eps = QTableWidgetItem(f"{len(batch.episodes)} eps ({ep_tags})")
            self._table_preview.setItem(row, 2, it_eps)

        plural = "file" if len(self._batches) == 1 else "files"
        self._lbl_plan_summary.setText(
            f"Will generate {len(self._batches)} combined video {plural} from {len(valid_pairs)} episode(s):"
        )

    def _on_start_combining_clicked(self) -> None:
        """Start async combination pipeline."""
        if not self._batches:
            return

        dest_dir = Path(self._txt_dest.text().strip()).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        self._is_combining = True
        self._cancelled = False
        self._btn_start.setEnabled(False)
        self._spin_batch_size.setEnabled(False)
        self._rb_mode_batch.setEnabled(False)
        self._rb_mode_all.setEnabled(False)
        self._rb_use_dubbed.setEnabled(False)
        self._rb_use_source.setEnabled(False)
        self._btn_browse.setEnabled(False)
        self._btn_cancel.setText("Stop")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(self._batches))
        self._progress_bar.setValue(0)
        self._btn_open_folder.setVisible(False)

        total_batches = len(self._batches)

        async def _combine_worker() -> tuple[int, list[str]]:
            succeeded_count = 0
            errors: list[str] = []

            for idx, batch in enumerate(self._batches, start=1):
                if self._cancelled:
                    break

                out_path = dest_dir / batch.output_filename
                logger.info(
                    "Combining {} videos into '{}' (batch {}/{})",
                    len(batch.input_paths),
                    out_path.name,
                    idx,
                    total_batches,
                )

                # Update UI status via bridge callback
                self._bridge.post_event(
                    lambda i=idx, b=batch: self._update_batch_progress(i, total_batches, b.output_filename)
                )

                code, _, stderr = await concatenate_videos(batch.input_paths, out_path)
                if code == 0 and out_path.exists() and out_path.stat().st_size > 0:
                    succeeded_count += 1
                else:
                    msg = f"Failed to combine {batch.output_filename}: {stderr[:150]}"
                    errors.append(msg)
                    logger.error(msg)

                self._bridge.post_event(lambda i=idx: self._progress_bar.setValue(i))

            return succeeded_count, errors

        def _on_success(result: tuple[int, list[str]]) -> None:
            succeeded_count, errors = result
            self._is_combining = False
            self._btn_cancel.setText("Close")
            self._btn_open_folder.setVisible(True)

            if self._cancelled:
                self._lbl_status.setText(f"Stopped. {succeeded_count} of {total_batches} file(s) created.")
                self._lbl_status.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 12px;")
            elif errors:
                self._lbl_status.setText(
                    f"Completed with issues: {succeeded_count}/{total_batches} combined. Error: {errors[0]}"
                )
                self._lbl_status.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 12px;")
            else:
                self._lbl_status.setText(
                    f"Successfully combined all {succeeded_count} video file(s) into output directory!"
                )
                self._lbl_status.setStyleSheet("color: #16a34a; font-weight: 600; font-size: 12px;")

        def _on_error(exc: Exception) -> None:
            self._is_combining = False
            self._btn_cancel.setText("Close")
            self._lbl_status.setText(f"Combination failed: {exc}")
            self._lbl_status.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 12px;")
            logger.error("Combination process error: {}", exc)

        self._bridge.run_async(_combine_worker(), on_success=_on_success, on_error=_on_error)

    def _update_batch_progress(self, current: int, total: int, filename: str) -> None:
        self._lbl_status.setText(f"Combining {current}/{total}: {filename}...")
        self._lbl_status.setStyleSheet("color: #2563eb; font-weight: 600; font-size: 12px;")

    def _on_cancel_clicked(self) -> None:
        if self._is_combining:
            self._cancelled = True
            kill_all_ffmpeg_processes()
            self._lbl_status.setText("Stopping combination...")
            return
        self.accept()

    def _on_open_output_folder_clicked(self) -> None:
        dest = Path(self._txt_dest.text().strip()).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        import subprocess

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(dest)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(dest)])
            else:
                subprocess.Popen(["xdg-open", str(dest)])
        except Exception as exc:
            logger.error("Failed to open output directory {}: {}", dest, exc)
