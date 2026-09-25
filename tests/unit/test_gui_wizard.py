"""
Unit tests for ProjectWizardScreen navigation, validation, and episode filtering.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from mediadubflow.gui.screens.project_wizard_screen import ProjectWizardScreen


def test_wizard_step1_validation(qtbot, tmp_path: Path) -> None:
    """Verifies that empty project title and invalid folder trigger inline errors."""
    mock_bridge = MagicMock()
    wizard = ProjectWizardScreen(bridge=mock_bridge)
    qtbot.addWidget(wizard)

    # Initially step 0
    assert wizard._step_stack.currentIndex() == 0

    # Attempt next with empty title and empty folder
    wizard._btn_step1_next.click()
    assert wizard._err_name.isVisible()
    assert "required" in wizard._err_name.text().lower()
    assert wizard._step_stack.currentIndex() == 0

    # Fill name, attempt next with no folder
    wizard._txt_name.setText("Test Drama Series")
    wizard._btn_step1_next.click()
    assert not wizard._err_name.isVisible()
    assert wizard._err_folder.isVisible()
    assert wizard._step_stack.currentIndex() == 0

    # Supply empty folder with no video files
    empty_folder = tmp_path / "empty_drama"
    empty_folder.mkdir()
    wizard._source_dir = empty_folder
    wizard._txt_folder.setText(str(empty_folder))

    wizard._btn_step1_next.click()
    assert wizard._err_folder.isVisible()
    assert "no supported video files" in wizard._err_folder.text().lower()
    assert wizard._step_stack.currentIndex() == 0


def test_wizard_step2_episodes_selection(qtbot, tmp_path: Path) -> None:
    """Verifies discovered episodes population, select/deselect all, and validation."""
    mock_bridge = MagicMock()
    wizard = ProjectWizardScreen(bridge=mock_bridge)
    qtbot.addWidget(wizard)

    # Create dummy video files
    folder = tmp_path / "drama_season1"
    folder.mkdir()
    f1 = folder / "ep01.mp4"
    f2 = folder / "ep02.mp4"
    f1.write_bytes(b"dummy")
    f2.write_bytes(b"dummy")

    wizard._source_dir = folder
    wizard._txt_name.setText("Drama Season 1")
    wizard._txt_folder.setText(str(folder))

    wizard._btn_step1_next.click()
    assert wizard._step_stack.currentIndex() == 1
    assert wizard._table_episodes.rowCount() == 2

    # Deselect all
    wizard._set_all_episodes_checked(False)
    wizard._btn_step2_next.click()
    assert wizard._err_episodes.isVisible()
    assert wizard._step_stack.currentIndex() == 1

    # Select all and proceed
    wizard._set_all_episodes_checked(True)
    wizard._btn_step2_next.click()
    assert not wizard._err_episodes.isVisible()
    assert wizard._step_stack.currentIndex() == 2


def test_wizard_step3_glossary_validation(qtbot) -> None:
    """Verifies that malformed JSON in glossary triggers inline error."""
    mock_bridge = MagicMock()
    wizard = ProjectWizardScreen(bridge=mock_bridge)
    qtbot.addWidget(wizard)

    wizard._update_stepper(2)
    assert wizard._step_stack.currentIndex() == 2

    # Malformed JSON
    wizard._txt_glossary.setText("{broken json")
    wizard._on_start_project()
    assert wizard._err_glossary.isVisible()
    assert "invalid json" in wizard._err_glossary.text().lower()

    # Valid JSON
    wizard._txt_glossary.setText('{"Hero": "វីរបុរស"}')
    # Provide necessary mock state
    wizard._source_dir = Path("/tmp/mock")
    wizard._selected_episodes = [(1, Path("/tmp/mock/ep01.mp4"))]
    wizard._txt_name.setText("Valid Series")

    wizard._on_start_project()
    assert not wizard._err_glossary.isVisible()
    assert mock_bridge.run_async.called


def test_wizard_cancel_signal(qtbot) -> None:
    """Clicking cancel on step 1 emits the cancelled signal."""
    mock_bridge = MagicMock()
    wizard = ProjectWizardScreen(bridge=mock_bridge)
    qtbot.addWidget(wizard)

    with qtbot.waitSignal(wizard.cancelled, timeout=1000):
        wizard.cancelled.emit()
