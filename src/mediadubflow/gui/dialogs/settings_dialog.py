"""
Settings Dialog — Application configuration and API key management.

Provides controls to update:
- Translation provider and API keys (OpenAI / Anthropic).
- Speech-to-text Whisper model size.
- Hardware compute device and concurrent episode worker limit.
"""

from __future__ import annotations

from pathlib import Path

import dotenv
from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config.settings import (
    PipelineOutputMode,
    TranslationProvider,
    settings,
)


class SettingsDialog(QDialog):
    """Modal dialog allowing users to configure runtime engine parameters and API keys."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings — MediaDubFlow")
        self.setMinimumWidth(500)
        self.setModal(True)
        self._setup_ui()
        self._load_current_values()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        title = QLabel("Engine & Localization Settings")
        title.setObjectName("sectionTitle")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0f172a;")
        main_layout.addWidget(title)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setSpacing(12)

        # 0. Workflow Mode
        self._cmb_output_mode = QComboBox()
        self._cmb_output_mode.addItem(
            "Voice Dubbing Only (No Subtitles)", PipelineOutputMode.VOICE_DUBBING_ONLY
        )
        self._cmb_output_mode.addItem(
            "Subtitles Only (No Voice Dubbing)", PipelineOutputMode.SUBTITLES_ONLY
        )
        self._cmb_output_mode.addItem(
            "Both (Voice Dubbing + Subtitles)", PipelineOutputMode.BOTH
        )
        form_layout.addRow("Workflow Mode:", self._cmb_output_mode)

        # 1. Translation Provider
        self._cmb_provider = QComboBox()
        self._cmb_provider.addItem("OpenAI (GPT-4o)", TranslationProvider.OPENAI)
        self._cmb_provider.addItem("Anthropic (Claude 3.5 Sonnet)", TranslationProvider.ANTHROPIC)
        self._cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        form_layout.addRow("Translation Provider:", self._cmb_provider)

        # 2. OpenAI API Key
        self._txt_openai_key = QLineEdit()
        self._txt_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_openai_key.setPlaceholderText("sk-...")
        form_layout.addRow("OpenAI API Key:", self._txt_openai_key)

        # 3. Anthropic API Key
        self._txt_anthropic_key = QLineEdit()
        self._txt_anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_anthropic_key.setPlaceholderText("sk-ant-...")
        form_layout.addRow("Anthropic API Key:", self._txt_anthropic_key)

        # 4. Whisper Model Size
        self._cmb_whisper = QComboBox()
        for size in ("tiny", "base", "small", "medium", "large-v3"):
            self._cmb_whisper.addItem(size, size)
        form_layout.addRow("Whisper STT Model:", self._cmb_whisper)

        # 5. Compute Device
        self._cmb_device = QComboBox()
        self._cmb_device.addItem("Auto-detect", "auto")
        self._cmb_device.addItem("CUDA (NVIDIA GPU)", "cuda")
        self._cmb_device.addItem("CPU", "cpu")
        form_layout.addRow("Compute Device:", self._cmb_device)

        # 6. Max Concurrent Episodes
        self._spin_concurrency = QSpinBox()
        self._spin_concurrency.setRange(1, 8)
        self._spin_concurrency.setValue(2)
        form_layout.addRow("Concurrent Episodes:", self._spin_concurrency)

        main_layout.addLayout(form_layout)

        # Status note
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #10b981; font-size: 12px;")
        self._lbl_status.setVisible(False)
        main_layout.addWidget(self._lbl_status)

        main_layout.addSpacing(8)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("Save Settings")
        btn_save.setObjectName("primaryButton")
        btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_save.clicked.connect(self._on_save_clicked)

        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        main_layout.addLayout(btn_layout)

    def _load_current_values(self) -> None:
        # Workflow Mode
        mode_idx = self._cmb_output_mode.findData(settings.output_mode)
        if mode_idx >= 0:
            self._cmb_output_mode.setCurrentIndex(mode_idx)

        # Provider
        idx = self._cmb_provider.findData(settings.translation_provider)
        if idx >= 0:
            self._cmb_provider.setCurrentIndex(idx)

        # Keys
        self._txt_openai_key.setText(settings.openai_api_key)
        self._txt_anthropic_key.setText(settings.anthropic_api_key)

        # Whisper
        whisper_idx = self._cmb_whisper.findData(settings.whisper_model_size)
        if whisper_idx >= 0:
            self._cmb_whisper.setCurrentIndex(whisper_idx)

        # Device
        dev_idx = self._cmb_device.findData(settings.device)
        if dev_idx >= 0:
            self._cmb_device.setCurrentIndex(dev_idx)

        # Concurrency
        self._spin_concurrency.setValue(settings.max_concurrent_episodes)
        self._on_provider_changed()

    def _on_provider_changed(self) -> None:
        provider = self._cmb_provider.currentData()
        if provider == TranslationProvider.OPENAI:
            self._txt_openai_key.setEnabled(True)
            self._txt_anthropic_key.setEnabled(False)
        else:
            self._txt_openai_key.setEnabled(False)
            self._txt_anthropic_key.setEnabled(True)

    def _on_save_clicked(self) -> None:
        output_mode = self._cmb_output_mode.currentData()
        provider = self._cmb_provider.currentData()
        openai_key = self._txt_openai_key.text().strip()
        anthropic_key = self._txt_anthropic_key.text().strip()
        whisper_size = self._cmb_whisper.currentData()
        device = self._cmb_device.currentData()
        concurrency = self._spin_concurrency.value()

        # Update runtime settings
        settings.output_mode = output_mode
        settings.translation_provider = provider
        if openai_key:
            settings.openai_api_key = openai_key
        if anthropic_key:
            settings.anthropic_api_key = anthropic_key
        settings.whisper_model_size = whisper_size
        settings.device = device
        settings.max_concurrent_episodes = concurrency

        logger.info(
            "Settings updated: provider={}, device={}, concurrency={}",
            provider,
            device,
            concurrency,
        )

        # Persist modified settings to .env
        try:
            env_path = Path(".env").resolve()
            if not env_path.exists():
                env_path.touch()
            dotenv.set_key(str(env_path), "OUTPUT_MODE", str(output_mode))
            dotenv.set_key(str(env_path), "TRANSLATION_PROVIDER", str(provider))
            if openai_key:
                dotenv.set_key(str(env_path), "OPENAI_API_KEY", openai_key)
            if anthropic_key:
                dotenv.set_key(str(env_path), "ANTHROPIC_API_KEY", anthropic_key)
            dotenv.set_key(str(env_path), "WHISPER_MODEL_SIZE", str(whisper_size))
            dotenv.set_key(str(env_path), "DEVICE", str(device))
            dotenv.set_key(str(env_path), "MAX_CONCURRENT_EPISODES", str(concurrency))
            logger.info("Settings saved to .env at {}", env_path)
        except Exception as exc:
            logger.warning("Could not persist settings to .env: {}", exc)

        self.accept()
