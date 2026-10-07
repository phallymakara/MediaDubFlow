"""
Edit Project Dialog — Modal dialog for modifying project metadata.

Allows the operator to update:
- Project title
- Destination output folder (with native directory picker)
- Target language

Validation enforces non-empty project names with inline error reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from mediadubflow.models.orm import Project


class EditProjectDialog(QDialog):
    """
    Modal dialog allowing operators to edit project metadata.

    Enforces validation on project name and provides directory browsing
    for output destination.
    """

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Project — MediaDubFlow")
        self.setMinimumWidth(480)
        self.setModal(True)

        self._project = project
        self._updated_data: dict[str, Any] = {}

        self._setup_ui()
        self._populate_fields()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # 1. Dialog Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)

        title = QLabel("Edit Project Details")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0f172a;")

        subtitle = QLabel("Update the project title and destination export directory.")
        subtitle.setStyleSheet("font-size: 13px; color: #64748b;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)

        # 2. Form Layout
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        # Project Name input + inline error
        name_container = QWidget()
        name_vbox = QVBoxLayout(name_container)
        name_vbox.setContentsMargins(0, 0, 0, 0)
        name_vbox.setSpacing(4)

        self._txt_name = QLineEdit()
        self._txt_name.textChanged.connect(self._on_name_text_changed)

        self._lbl_name_error = QLabel()
        self._lbl_name_error.setStyleSheet("color: #dc2626; font-size: 11px;")
        self._lbl_name_error.setVisible(False)

        name_vbox.addWidget(self._txt_name)
        name_vbox.addWidget(self._lbl_name_error)

        form_layout.addRow("Project Name:", name_container)

        # Source Folder (Read-only reference)
        self._txt_source = QLineEdit()
        self._txt_source.setReadOnly(True)
        self._txt_source.setStyleSheet("color: #64748b; background-color: #f8fafc;")
        self._txt_source.setToolTip("Source directory cannot be modified once created.")
        form_layout.addRow("Source Folder:", self._txt_source)

        # Output Destination Folder + Browse Button
        out_container = QWidget()
        out_hbox = QHBoxLayout(out_container)
        out_hbox.setContentsMargins(0, 0, 0, 0)
        out_hbox.setSpacing(8)

        self._txt_output = QLineEdit()
        self._btn_browse_output = QPushButton("Browse...")
        self._btn_browse_output.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_browse_output.clicked.connect(self._on_browse_output_clicked)

        out_hbox.addWidget(self._txt_output, stretch=1)
        out_hbox.addWidget(self._btn_browse_output)

        form_layout.addRow("Output Folder:", out_container)

        # Target Language Dropdown
        self._cmb_target_lang = QComboBox()
        self._cmb_target_lang.addItem("Khmer (km)", "km")
        self._cmb_target_lang.addItem("English (en)", "en")
        self._cmb_target_lang.addItem("Chinese (zh)", "zh")
        self._cmb_target_lang.addItem("Korean (ko)", "ko")
        self._cmb_target_lang.addItem("Japanese (ja)", "ja")
        form_layout.addRow("Target Language:", self._cmb_target_lang)

        main_layout.addLayout(form_layout)

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cancel.clicked.connect(self.reject)

        self._btn_save = QPushButton("Save Changes")
        self._btn_save.setObjectName("primaryButton")
        self._btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_save.clicked.connect(self._on_save_clicked)

        btn_layout.addWidget(self._btn_cancel)
        btn_layout.addWidget(self._btn_save)

        main_layout.addSpacing(8)
        main_layout.addLayout(btn_layout)

    def _populate_fields(self) -> None:
        """Populate form inputs from the project model instance."""
        self._txt_name.setText(self._project.name)
        self._txt_source.setText(self._project.source_folder)
        self._txt_output.setText(self._project.output_folder)

        idx = self._cmb_target_lang.findData(self._project.target_language)
        if idx >= 0:
            self._cmb_target_lang.setCurrentIndex(idx)
        else:
            self._cmb_target_lang.addItem(
                f"{self._project.target_language.upper()}",
                self._project.target_language,
            )
            self._cmb_target_lang.setCurrentIndex(self._cmb_target_lang.count() - 1)

    def _on_name_text_changed(self) -> None:
        """Clear inline error once operator begins typing."""
        if self._lbl_name_error.isVisible():
            self._lbl_name_error.setVisible(False)
            self._txt_name.setProperty("hasError", False)
            self._txt_name.style().unpolish(self._txt_name)
            self._txt_name.style().polish(self._txt_name)

    def _on_browse_output_clicked(self) -> None:
        """Open directory picker to select a custom output folder."""
        initial_dir = self._txt_output.text() or self._project.source_folder
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Output Destination Folder",
            initial_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if chosen:
            self._txt_output.setText(str(Path(chosen).resolve()))

    def _on_save_clicked(self) -> None:
        """Validate inputs and accept dialog if valid."""
        clean_name = self._txt_name.text().strip()
        if not clean_name:
            self._lbl_name_error.setText("Project name cannot be empty.")
            self._lbl_name_error.setVisible(True)
            self._txt_name.setProperty("hasError", True)
            self._txt_name.style().unpolish(self._txt_name)
            self._txt_name.style().polish(self._txt_name)
            self._txt_name.setFocus()
            return

        out_folder = self._txt_output.text().strip() or self._project.output_folder
        target_lang = self._cmb_target_lang.currentData() or "km"

        self._updated_data = {
            "name": clean_name,
            "output_folder": out_folder,
            "target_language": target_lang,
        }
        self.accept()

    def get_updated_values(self) -> dict[str, Any]:
        """Return the dictionary of validated updated values."""
        return self._updated_data
