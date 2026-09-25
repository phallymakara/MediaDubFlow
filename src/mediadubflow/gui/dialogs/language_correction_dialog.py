"""
Language Correction Dialog — User override for low-confidence language detection.

Prompted when automatic Whisper audio probing yields confidence below 0.85.
Allows the operator to confirm the detected language or manually select
the correct spoken language before proceeding with transcription and translation.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LanguageCorrectionDialog(QDialog):
    """
    Modal dialog prompted when language detection confidence is low.

    Returns:
        The verified 2-letter ISO language code (e.g., 'zh', 'ko', 'en') upon accept.
    """

    def __init__(
        self,
        episode_number: int,
        detected_language: str,
        confidence: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Verify Spoken Language — MediaDubFlow")
        self.setMinimumWidth(440)
        self.setModal(True)

        self._episode_number = episode_number
        self._detected_language = detected_language
        self._confidence = confidence
        self._selected_language = detected_language

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # Title
        title = QLabel(f"Verify Language: Episode {self._episode_number:02d}")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0f172a;")
        main_layout.addWidget(title)

        # Confidence Note
        pct_conf = int(self._confidence * 100)
        note_text = (
            f"Automatic audio probing detected <b>{self._detected_language.upper()}</b> "
            f"with lower confidence (<b>{pct_conf}%</b>).<br>"
            "Please confirm or select the correct spoken language for accurate transcription."
        )
        note_lbl = QLabel(note_text)
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(
            "background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; "
            "border-radius: 4px; padding: 10px; font-size: 13px;"
        )
        main_layout.addWidget(note_lbl)

        main_layout.addSpacing(6)

        # Language Selector
        sel_lbl = QLabel("Select Spoken Language:")
        sel_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        main_layout.addWidget(sel_lbl)

        self._cmb_lang = QComboBox()
        languages = [
            ("Chinese (Mandarin / Yue)", "zh"),
            ("Korean (ko)", "ko"),
            ("English (en)", "en"),
            ("Japanese (ja)", "ja"),
            ("Thai (th)", "th"),
            ("Vietnamese (vi)", "vi"),
            ("Spanish (es)", "es"),
            ("French (fr)", "fr"),
        ]
        for name, code in languages:
            self._cmb_lang.addItem(name, code)

        # Pre-select detected language if in list
        idx = self._cmb_lang.findData(self._detected_language.lower())
        if idx >= 0:
            self._cmb_lang.setCurrentIndex(idx)

        main_layout.addWidget(self._cmb_lang)

        main_layout.addSpacing(12)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        btn_keep = QPushButton(f"Keep '{self._detected_language.upper()}'")
        btn_keep.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_keep.clicked.connect(self._on_keep_detected)

        btn_confirm = QPushButton("Confirm Language")
        btn_confirm.setObjectName("primaryButton")
        btn_confirm.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_confirm.clicked.connect(self._on_confirm_selected)

        btn_layout.addWidget(btn_keep)
        btn_layout.addWidget(btn_confirm)
        main_layout.addLayout(btn_layout)

    def _on_keep_detected(self) -> None:
        self._selected_language = self._detected_language
        self.accept()

    def _on_confirm_selected(self) -> None:
        self._selected_language = self._cmb_lang.currentData()
        self.accept()

    def get_selected_language(self) -> str:
        """Return the confirmed 2-letter ISO language code."""
        return self._selected_language
