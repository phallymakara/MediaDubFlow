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

    # Verify loaded values
    assert dlg._cmb_provider.count() == 2
    assert dlg._spin_concurrency.value() >= 1

    # Switch values
    dlg._spin_concurrency.setValue(4)
    dlg._txt_openai_key.setText("sk-test-key-mock")

    # Trigger save
    dlg._on_save_clicked()

    assert settings.max_concurrent_episodes == 4
    assert settings.openai_api_key == "sk-test-key-mock"


def test_settings_dialog_provider_toggle(qtbot) -> None:
    """Verifies switching provider disables/enables corresponding API key input."""
    dlg = SettingsDialog()
    qtbot.addWidget(dlg)

    # Set to OpenAI
    idx_openai = dlg._cmb_provider.findData(TranslationProvider.OPENAI)
    dlg._cmb_provider.setCurrentIndex(idx_openai)
    assert dlg._txt_openai_key.isEnabled()
    assert not dlg._txt_anthropic_key.isEnabled()

    # Set to Anthropic
    idx_anthropic = dlg._cmb_provider.findData(TranslationProvider.ANTHROPIC)
    dlg._cmb_provider.setCurrentIndex(idx_anthropic)
    assert not dlg._txt_openai_key.isEnabled()
    assert dlg._txt_anthropic_key.isEnabled()


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
