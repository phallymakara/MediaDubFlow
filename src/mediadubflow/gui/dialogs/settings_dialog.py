"""
Settings Dialog — Application configuration and API key management.

Provides controls to update:
- Translation provider with required endpoints, API keys, and models
  (OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude).
- Speech-to-text Whisper model size.
- Hardware compute device and concurrent episode worker limit.
"""

from __future__ import annotations

from pathlib import Path

import dotenv
from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mediadubflow.config.settings import (
    TranslationProvider,
    settings,
)


class SettingsDialog(QDialog):
    """Modal dialog allowing users to configure runtime engine parameters and API keys."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings — MediaDubFlow")
        self.setMinimumSize(500, 420)
        self.resize(560, 600)
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

        # Scroll Area for Form Content (smooth scrolling for Windows and Mac users)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.verticalScrollBar().setSingleStep(16)

        form_container = QWidget()
        form_container_layout = QVBoxLayout(form_container)
        form_container_layout.setContentsMargins(0, 4, 8, 4)
        form_container_layout.setSpacing(12)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setSpacing(12)

        # 1. Translation Provider
        self._cmb_provider = QComboBox()
        self._cmb_provider.addItem("OpenAI (GPT-4o / Compatible)", TranslationProvider.OPENAI)
        self._cmb_provider.addItem("Azure OpenAI (Custom Endpoint)", TranslationProvider.AZURE_OPENAI)
        self._cmb_provider.addItem("Google Gemini (Gemini 2.0 Flash)", TranslationProvider.GEMINI)
        self._cmb_provider.addItem("Anthropic (Claude 3.5 Sonnet)", TranslationProvider.ANTHROPIC)
        self._cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        form_layout.addRow("Translation Provider:", self._cmb_provider)

        form_container_layout.addLayout(form_layout)

        # 2. Dynamic Provider Configuration Stack
        self._provider_stack = QStackedWidget()

        # Page A: OpenAI
        self._page_openai = QWidget()
        layout_openai = QFormLayout(self._page_openai)
        layout_openai.setContentsMargins(0, 0, 0, 0)
        layout_openai.setSpacing(10)
        self._txt_openai_key = QLineEdit()
        self._txt_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_openai_key.setPlaceholderText("sk-...")
        layout_openai.addRow("OpenAI API Key:", self._txt_openai_key)

        self._txt_openai_model = QLineEdit()
        self._txt_openai_model.setPlaceholderText("gpt-4o")
        layout_openai.addRow("Model Name:", self._txt_openai_model)

        self._txt_openai_base_url = QLineEdit()
        self._txt_openai_base_url.setPlaceholderText("https://api.openai.com/v1 (optional)")
        layout_openai.addRow("Endpoint / Base URL:", self._txt_openai_base_url)

        # Page B: Azure OpenAI
        self._page_azure = QWidget()
        layout_azure = QFormLayout(self._page_azure)
        layout_azure.setContentsMargins(0, 0, 0, 0)
        layout_azure.setSpacing(10)
        self._txt_azure_endpoint = QLineEdit()
        self._txt_azure_endpoint.setPlaceholderText("https://<resource-name>.openai.azure.com/")
        layout_azure.addRow("Azure Endpoint URL:", self._txt_azure_endpoint)

        self._txt_azure_key = QLineEdit()
        self._txt_azure_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_azure_key.setPlaceholderText("Enter Azure OpenAI API key...")
        layout_azure.addRow("Azure API Key:", self._txt_azure_key)

        self._txt_azure_deployment = QLineEdit()
        self._txt_azure_deployment.setPlaceholderText("e.g. gpt-4o")
        layout_azure.addRow("Deployment Name:", self._txt_azure_deployment)

        self._txt_azure_version = QLineEdit()
        self._txt_azure_version.setPlaceholderText("2024-08-01-preview")
        layout_azure.addRow("API Version:", self._txt_azure_version)

        # Page C: Google Gemini
        self._page_gemini = QWidget()
        layout_gemini = QFormLayout(self._page_gemini)
        layout_gemini.setContentsMargins(0, 0, 0, 0)
        layout_gemini.setSpacing(10)
        self._txt_gemini_key = QLineEdit()
        self._txt_gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_gemini_key.setPlaceholderText("AIzaSy...")
        layout_gemini.addRow("Google Gemini API Key:", self._txt_gemini_key)

        self._txt_gemini_model = QLineEdit()
        self._txt_gemini_model.setPlaceholderText("gemini-3.8-flash")
        layout_gemini.addRow("Model Name:", self._txt_gemini_model)

        # Page D: Anthropic
        self._page_anthropic = QWidget()
        layout_anthropic = QFormLayout(self._page_anthropic)
        layout_anthropic.setContentsMargins(0, 0, 0, 0)
        layout_anthropic.setSpacing(10)
        self._txt_anthropic_key = QLineEdit()
        self._txt_anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._txt_anthropic_key.setPlaceholderText("sk-ant-...")
        layout_anthropic.addRow("Anthropic API Key:", self._txt_anthropic_key)

        self._txt_anthropic_model = QLineEdit()
        self._txt_anthropic_model.setPlaceholderText("claude-3-5-sonnet-20241022")
        layout_anthropic.addRow("Model Name:", self._txt_anthropic_model)

        self._txt_anthropic_base_url = QLineEdit()
        self._txt_anthropic_base_url.setPlaceholderText("https://api.anthropic.com (optional)")
        layout_anthropic.addRow("Endpoint / Base URL:", self._txt_anthropic_base_url)

        self._provider_stack.addWidget(self._page_openai)
        self._provider_stack.addWidget(self._page_azure)
        self._provider_stack.addWidget(self._page_gemini)
        self._provider_stack.addWidget(self._page_anthropic)

        form_container_layout.addWidget(self._provider_stack)

        # General Engine Configuration
        general_form_layout = QFormLayout()
        general_form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        general_form_layout.setSpacing(12)

        # Whisper Model Size
        self._cmb_whisper = QComboBox()
        for size in ("tiny", "base", "small", "medium", "large-v3"):
            self._cmb_whisper.addItem(size, size)
        general_form_layout.addRow("Whisper STT Model:", self._cmb_whisper)

        # Compute Device
        self._cmb_device = QComboBox()
        self._cmb_device.addItem("Auto-detect", "auto")
        self._cmb_device.addItem("CUDA (NVIDIA GPU)", "cuda")
        self._cmb_device.addItem("CPU", "cpu")
        general_form_layout.addRow("Compute Device:", self._cmb_device)

        # Max Concurrent Episodes
        self._spin_concurrency = QSpinBox()
        self._spin_concurrency.setRange(1, 8)
        self._spin_concurrency.setValue(2)
        general_form_layout.addRow("Concurrent Episodes:", self._spin_concurrency)

        # TTS & Audio Dubbing Controls
        tts_header = QLabel("TTS Voice Dubbing & Audio Mixing")
        tts_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a; margin-top: 14px;")
        general_form_layout.addRow(tts_header)

        # Mute Original Speaker Checkbox
        self._chk_mute_original = QCheckBox("Mute Original Voice (Silences original speaker during dialogue, keeps background sound)")
        self._chk_mute_original.setWordWrap(True)
        self._chk_mute_original.setStyleSheet("font-weight: 600; font-size: 13px; color: #0f172a;")
        general_form_layout.addRow("", self._chk_mute_original)

        # TTS Voice Selection
        self._cmb_tts_voice = QComboBox()
        self._cmb_tts_voice.addItem("Piseth (Male, Khmer Natural) — km-KH-PisethNeural", "km-KH-PisethNeural")
        self._cmb_tts_voice.addItem("Sreymom (Female, Khmer Natural) — km-KH-SreymomNeural", "km-KH-SreymomNeural")
        general_form_layout.addRow("TTS Voice:", self._cmb_tts_voice)

        # Sound Volume Control (0% to 200%)
        vol_box = QHBoxLayout()
        vol_box.setSpacing(8)
        self._slider_volume = QSlider(Qt.Orientation.Horizontal)
        self._slider_volume.setRange(0, 200)
        self._slider_volume.setSingleStep(5)
        self._slider_volume.setPageStep(10)
        self._slider_volume.setToolTip("Horizontal scroll or drag to increase/decrease TTS sound volume")
        self._lbl_volume_val = QLabel("120%")
        self._lbl_volume_val.setFixedWidth(44)
        self._lbl_volume_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb;")
        self._slider_volume.valueChanged.connect(lambda v: self._lbl_volume_val.setText(f"{v}%"))
        vol_box.addWidget(self._slider_volume, stretch=1)
        vol_box.addWidget(self._lbl_volume_val)
        general_form_layout.addRow("TTS Sound Volume:", vol_box)

        # Background Sound Volume Control (0% to 150%)
        bg_vol_box = QHBoxLayout()
        bg_vol_box.setSpacing(8)
        self._slider_bg_volume = QSlider(Qt.Orientation.Horizontal)
        self._slider_bg_volume.setRange(0, 150)
        self._slider_bg_volume.setSingleStep(5)
        self._slider_bg_volume.setPageStep(10)
        self._slider_bg_volume.setToolTip("Horizontal scroll or drag to increase/decrease background sound volume")
        self._lbl_bg_vol_val = QLabel("80%")
        self._lbl_bg_vol_val.setFixedWidth(44)
        self._lbl_bg_vol_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb;")
        self._slider_bg_volume.valueChanged.connect(lambda v: self._lbl_bg_vol_val.setText(f"{v}%"))
        bg_vol_box.addWidget(self._slider_bg_volume, stretch=1)
        bg_vol_box.addWidget(self._lbl_bg_vol_val)
        general_form_layout.addRow("Background Sound Volume:", bg_vol_box)

        # Speech Speed Horizontal Slider (0.50x to 2.00x)
        speed_box = QHBoxLayout()
        speed_box.setSpacing(8)
        self._slider_speed = QSlider(Qt.Orientation.Horizontal)
        self._slider_speed.setRange(50, 200)
        self._slider_speed.setSingleStep(5)
        self._slider_speed.setPageStep(10)
        self._slider_speed.setToolTip("Horizontal scroll or drag to increase/decrease speech speed (0.50x to 2.00x)")
        self._lbl_speed_val = QLabel("1.00x")
        self._lbl_speed_val.setFixedWidth(44)
        self._lbl_speed_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb;")
        self._slider_speed.valueChanged.connect(lambda v: self._lbl_speed_val.setText(f"{v/100:.2f}x"))
        speed_box.addWidget(self._slider_speed, stretch=1)
        speed_box.addWidget(self._lbl_speed_val)
        general_form_layout.addRow("Speech Speed:", speed_box)

        form_container_layout.addLayout(general_form_layout)

        # Status note
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #10b981; font-size: 12px;")
        self._lbl_status.setVisible(False)
        form_container_layout.addWidget(self._lbl_status)

        form_container_layout.addStretch(1)
        scroll_area.setWidget(form_container)
        main_layout.addWidget(scroll_area, stretch=1)

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

    @staticmethod
    def _clean_val(val: str | None) -> str:
        """Strip dummy template placeholders and return empty string if not a real user input."""
        s = (val or "").strip()
        if (
            not s
            or s.endswith("...")
            or "<" in s
            or ">" in s
            or s in ("sk-...", "sk-ant-...", "AIzaSy...", "hf_...", "none", "null")
        ):
            return ""
        return s

    def _load_current_values(self) -> None:
        """Load settings into GUI inputs, ensuring unconfigured fields start completely empty."""
        # 1. Provider selection
        idx = self._cmb_provider.findData(settings.translation_provider)
        if idx >= 0:
            self._cmb_provider.setCurrentIndex(idx)

        # 2. Populate Provider fields (cleanly empty if user hasn't input them yet)
        self._txt_openai_key.setText(self._clean_val(settings.openai_api_key))
        self._txt_openai_model.setText(self._clean_val(settings.openai_model) or "gpt-4o")
        self._txt_openai_base_url.setText(self._clean_val(settings.openai_base_url))

        self._txt_azure_endpoint.setText(self._clean_val(settings.azure_openai_endpoint))
        self._txt_azure_key.setText(self._clean_val(settings.azure_openai_api_key))
        self._txt_azure_deployment.setText(self._clean_val(settings.azure_openai_deployment_name) or "gpt-4o")
        self._txt_azure_version.setText(self._clean_val(settings.azure_openai_api_version) or "2024-08-01-preview")

        self._txt_gemini_key.setText(self._clean_val(settings.gemini_api_key))
        self._txt_gemini_model.setText(self._clean_val(settings.gemini_model) or "gemini-2.0-flash")

        self._txt_anthropic_key.setText(self._clean_val(settings.anthropic_api_key))
        self._txt_anthropic_model.setText(self._clean_val(settings.anthropic_model) or "claude-3-5-sonnet-20241022")
        self._txt_anthropic_base_url.setText(self._clean_val(settings.anthropic_base_url))

        # 3. Whisper
        whisper_idx = self._cmb_whisper.findData(settings.whisper_model_size)
        if whisper_idx >= 0:
            self._cmb_whisper.setCurrentIndex(whisper_idx)

        # 4. Device
        dev_idx = self._cmb_device.findData(settings.device)
        if dev_idx >= 0:
            self._cmb_device.setCurrentIndex(dev_idx)

        # 5. Concurrency
        self._spin_concurrency.setValue(settings.max_concurrent_episodes)

        # 6. TTS Audio & Speed Settings
        voice_idx = self._cmb_tts_voice.findData(settings.tts_voice)
        if voice_idx >= 0:
            self._cmb_tts_voice.setCurrentIndex(voice_idx)

        self._chk_mute_original.setChecked(settings.mute_original_speaker)
        vol_val = int(round(settings.tts_volume * 100))
        self._slider_volume.setValue(vol_val)
        self._lbl_volume_val.setText(f"{vol_val}%")

        bg_vol_val = int(round(settings.original_audio_volume * 100))
        self._slider_bg_volume.setValue(bg_vol_val)
        self._lbl_bg_vol_val.setText(f"{bg_vol_val}%")

        speed_val = int(round(settings.tts_speed * 100))
        self._slider_speed.setValue(speed_val)
        self._lbl_speed_val.setText(f"{settings.tts_speed:.2f}x")

        self._on_provider_changed()

    def _on_provider_changed(self) -> None:
        """Switch provider config stack widget dynamically based on chosen provider."""
        provider = self._cmb_provider.currentData()
        if provider == TranslationProvider.OPENAI:
            self._provider_stack.setCurrentWidget(self._page_openai)
        elif provider == TranslationProvider.AZURE_OPENAI:
            self._provider_stack.setCurrentWidget(self._page_azure)
        elif provider == TranslationProvider.GEMINI:
            self._provider_stack.setCurrentWidget(self._page_gemini)
        elif provider == TranslationProvider.ANTHROPIC:
            self._provider_stack.setCurrentWidget(self._page_anthropic)

    def _on_save_clicked(self) -> None:
        provider = self._cmb_provider.currentData()
        whisper_size = self._cmb_whisper.currentData()
        device = self._cmb_device.currentData()
        concurrency = self._spin_concurrency.value()

        # Sanitize all provider fields
        openai_key = self._clean_val(self._txt_openai_key.text())
        openai_model = self._clean_val(self._txt_openai_model.text()) or "gpt-4o"
        openai_base_url = self._clean_val(self._txt_openai_base_url.text())

        azure_endpoint = self._clean_val(self._txt_azure_endpoint.text())
        azure_key = self._clean_val(self._txt_azure_key.text())
        azure_deployment = self._clean_val(self._txt_azure_deployment.text()) or "gpt-4o"
        azure_version = self._clean_val(self._txt_azure_version.text()) or "2024-08-01-preview"

        gemini_key = self._clean_val(self._txt_gemini_key.text())
        gemini_model = self._clean_val(self._txt_gemini_model.text()) or "gemini-2.0-flash"

        anthropic_key = self._clean_val(self._txt_anthropic_key.text())
        anthropic_model = self._clean_val(self._txt_anthropic_model.text()) or "claude-3-5-sonnet-20241022"
        anthropic_base_url = self._clean_val(self._txt_anthropic_base_url.text())

        # Update runtime settings
        settings.translation_provider = provider

        settings.openai_api_key = openai_key
        settings.openai_model = openai_model
        settings.openai_base_url = openai_base_url

        settings.azure_openai_endpoint = azure_endpoint
        settings.azure_openai_api_key = azure_key
        settings.azure_openai_deployment_name = azure_deployment
        settings.azure_openai_api_version = azure_version

        settings.gemini_api_key = gemini_key
        settings.gemini_model = gemini_model

        settings.anthropic_api_key = anthropic_key
        settings.anthropic_model = anthropic_model
        settings.anthropic_base_url = anthropic_base_url

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
            dotenv.set_key(str(env_path), "TRANSLATION_PROVIDER", str(provider))

            dotenv.set_key(str(env_path), "OPENAI_API_KEY", openai_key)
            dotenv.set_key(str(env_path), "OPENAI_MODEL", openai_model)
            dotenv.set_key(str(env_path), "OPENAI_BASE_URL", openai_base_url)

            dotenv.set_key(str(env_path), "AZURE_OPENAI_ENDPOINT", azure_endpoint)
            dotenv.set_key(str(env_path), "AZURE_OPENAI_API_KEY", azure_key)
            dotenv.set_key(str(env_path), "AZURE_OPENAI_DEPLOYMENT_NAME", azure_deployment)
            dotenv.set_key(str(env_path), "AZURE_OPENAI_API_VERSION", azure_version)

            dotenv.set_key(str(env_path), "GEMINI_API_KEY", gemini_key)
            dotenv.set_key(str(env_path), "GEMINI_MODEL", gemini_model)

            dotenv.set_key(str(env_path), "ANTHROPIC_API_KEY", anthropic_key)
            dotenv.set_key(str(env_path), "ANTHROPIC_MODEL", anthropic_model)
            dotenv.set_key(str(env_path), "ANTHROPIC_BASE_URL", anthropic_base_url)

            dotenv.set_key(str(env_path), "WHISPER_MODEL_SIZE", str(whisper_size))
            dotenv.set_key(str(env_path), "DEVICE", str(device))
            dotenv.set_key(str(env_path), "MAX_CONCURRENT_EPISODES", str(concurrency))
            dotenv.set_key(str(env_path), "TTS_VOICE", str(self._cmb_tts_voice.currentData()))
            dotenv.set_key(str(env_path), "TTS_SPEED", str(round(self._slider_speed.value() / 100.0, 2)))
            dotenv.set_key(str(env_path), "TTS_VOLUME", str(round(self._slider_volume.value() / 100.0, 2)))
            dotenv.set_key(str(env_path), "ORIGINAL_AUDIO_VOLUME", str(round(self._slider_bg_volume.value() / 100.0, 2)))
            dotenv.set_key(str(env_path), "MUTE_ORIGINAL_SPEAKER", "true" if self._chk_mute_original.isChecked() else "false")

            # Update settings object
            settings.tts_voice = str(self._cmb_tts_voice.currentData())
            settings.tts_speed = round(self._slider_speed.value() / 100.0, 2)
            settings.tts_volume = round(self._slider_volume.value() / 100.0, 2)
            settings.original_audio_volume = round(self._slider_bg_volume.value() / 100.0, 2)
            settings.mute_original_speaker = self._chk_mute_original.isChecked()

            logger.info("Settings saved to .env at {}", env_path)
        except Exception as exc:
            logger.warning("Could not persist settings to .env: {}", exc)

        self.accept()
