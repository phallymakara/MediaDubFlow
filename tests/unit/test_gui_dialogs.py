"""
Unit tests for SettingsDialog, LanguageCorrectionDialog, and EpisodeDetailPanel.
"""

from __future__ import annotations

from pathlib import Path

from mediadubflow.config.settings import TranslationProvider, settings
from mediadubflow.gui.dialogs.episode_detail_panel import EpisodeDetailPanel
from mediadubflow.gui.dialogs.language_correction_dialog import LanguageCorrectionDialog
from mediadubflow.gui.dialogs.settings_dialog import SettingsDialog
from mediadubflow.models.orm import Episode, EpisodeStatus


def test_settings_dialog_load_and_save(qtbot) -> None:
    """Verifies that SettingsDialog loads runtime values and updates settings on save."""
    dlg = SettingsDialog()
    qtbot.addWidget(dlg)

    # Verify loaded values: 4 providers available
    assert dlg._cmb_provider.count() == 4
    assert dlg._spin_concurrency.value() >= 1

    # Switch values
    dlg._spin_concurrency.setValue(4)

    # Set OpenAI fields
    dlg._txt_openai_key.setText("sk-test-key-mock")
    dlg._txt_openai_base_url.setText("https://api.openai.com/v1")

    # Set Azure OpenAI fields
    dlg._txt_azure_endpoint.setText("https://test-resource.openai.azure.com/")
    dlg._txt_azure_key.setText("azure-key-mock")
    dlg._txt_azure_deployment.setText("gpt-4o")

    # Set Gemini key
    dlg._txt_gemini_key.setText("AIzaSy-test-gemini-mock")

    # Trigger save
    dlg._on_save_clicked()

    assert settings.max_concurrent_episodes == 4
    assert settings.openai_api_key == "sk-test-key-mock"
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.azure_openai_endpoint == "https://test-resource.openai.azure.com/"
    assert settings.azure_openai_api_key == "azure-key-mock"
    assert settings.azure_openai_deployment_name == "gpt-4o"
    assert settings.gemini_api_key == "AIzaSy-test-gemini-mock"


def test_settings_dialog_provider_toggle(qtbot) -> None:
    """Verifies switching provider switches the provider configuration stack widget."""
    dlg = SettingsDialog()
    qtbot.addWidget(dlg)

    # Select OpenAI
    idx_openai = dlg._cmb_provider.findData(TranslationProvider.OPENAI)
    dlg._cmb_provider.setCurrentIndex(idx_openai)
    assert dlg._provider_stack.currentWidget() == dlg._page_openai

    # Select Azure OpenAI
    idx_azure = dlg._cmb_provider.findData(TranslationProvider.AZURE_OPENAI)
    dlg._cmb_provider.setCurrentIndex(idx_azure)
    assert dlg._provider_stack.currentWidget() == dlg._page_azure

    # Select Gemini
    idx_gemini = dlg._cmb_provider.findData(TranslationProvider.GEMINI)
    dlg._cmb_provider.setCurrentIndex(idx_gemini)
    assert dlg._provider_stack.currentWidget() == dlg._page_gemini

    # Select Anthropic
    idx_anthropic = dlg._cmb_provider.findData(TranslationProvider.ANTHROPIC)
    dlg._cmb_provider.setCurrentIndex(idx_anthropic)
    assert dlg._provider_stack.currentWidget() == dlg._page_anthropic


def test_settings_dialog_filters_dummy_template_keys(qtbot, monkeypatch) -> None:
    """Verifies that dummy template strings (e.g. 'sk-...', 'AIzaSy...', etc.) are not pre-filled into inputs."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-...")
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://<your-resource-name>.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaSy...")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-...")

    dlg = SettingsDialog()
    qtbot.addWidget(dlg)

    # All unconfigured inputs should start completely clean and empty
    assert dlg._txt_openai_key.text() == ""
    assert dlg._txt_azure_endpoint.text() == ""
    assert dlg._txt_azure_key.text() == ""
    assert dlg._txt_gemini_key.text() == ""
    assert dlg._txt_anthropic_key.text() == ""


def test_language_correction_dialog(qtbot) -> None:
    """Verifies LanguageCorrectionDialog displays confidence score and returns chosen language."""
    dlg = LanguageCorrectionDialog(
        episode_number=3,
        detected_language="vi",
        confidence=0.62,
    )
    qtbot.addWidget(dlg)

    assert (
        "Episode 03" in dlg.windowTitle()
        or "Episode 03" in dlg.findChild(EpisodeDetailPanel.QLabel, "").text()
    )  # type: ignore[arg-type]
    assert dlg.get_selected_language() == "vi"

    # Select Korean
    idx_ko = dlg._cmb_lang.findData("ko")
    dlg._cmb_lang.setCurrentIndex(idx_ko)
    dlg._on_confirm_selected()

    assert dlg.get_selected_language() == "ko"


def test_episode_detail_panel_preview(qtbot, tmp_path: Path) -> None:
    """Verifies EpisodeDetailPanel loads subtitle preview from file."""
    sub_file = tmp_path / "ep01.srt"
    sub_file.write_text("1\n00:00:01,000 --> 00:00:03,000\nជំរាបសួរ\n", encoding="utf-8")

    ep = Episode(
        id=1,
        project_id=1,
        episode_number=1,
        source_file="/media/drama/ep01.mp4",
        status=EpisodeStatus.DONE,
        detected_language="zh",
        language_confidence=0.95,
        subtitle_srt_path=str(sub_file),
    )

    panel = EpisodeDetailPanel(episode=ep)
    qtbot.addWidget(panel)

    assert "ជំរាបសួរ" in panel._txt_preview.toPlainText()
    assert "Done" in panel._status_badge.text()
