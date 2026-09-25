"""
Unit tests for WorkspaceScreen live queue, JobManager signal handling, and KPIs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mediadubflow.gui.screens.workspace_screen import WorkspaceScreen
from mediadubflow.models.orm import Episode, EpisodeStatus, Project


def test_workspace_initial_state(qtbot) -> None:
    """Verifies that WorkspaceScreen initializes with clean controls and headers."""
    mock_bridge = MagicMock()
    mock_jm = MagicMock()

    screen = WorkspaceScreen(bridge=mock_bridge, job_manager=mock_jm)
    qtbot.addWidget(screen)

    assert "No Project Loaded" in screen._lbl_project_title.text()
    assert screen._overall_progress.value() == 0
    assert screen._table.rowCount() == 0
    assert "Ready" in screen._lbl_activity_ticker.text()


def test_workspace_project_population(qtbot) -> None:
    """Verifies table rows and KPI metrics are calculated when a project is loaded."""
    mock_bridge = MagicMock()
    mock_jm = MagicMock()

    screen = WorkspaceScreen(bridge=mock_bridge, job_manager=mock_jm)
    qtbot.addWidget(screen)

    # Prepare mock Project with 3 episodes: 1 done, 1 pending, 1 failed
    project = Project(
        id=10,
        name="Ancient Love",
        source_folder="/media/drama",
        output_folder="/media/output",
    )
    ep1 = Episode(
        id=101,
        project_id=10,
        episode_number=1,
        source_file="/media/drama/ep01.mp4",
        status=EpisodeStatus.DONE,
    )
    ep2 = Episode(
        id=102,
        project_id=10,
        episode_number=2,
        source_file="/media/drama/ep02.mp4",
        status=EpisodeStatus.PENDING,
    )
    ep3 = Episode(
        id=103,
        project_id=10,
        episode_number=3,
        source_file="/media/drama/ep03.mp4",
        status=EpisodeStatus.FAILED,
    )
    project.episodes = [ep1, ep2, ep3]

    screen._on_project_loaded(project)

    assert screen._lbl_project_title.text() == "Ancient Love"
    assert screen._table.rowCount() == 3
    assert screen._overall_progress.value() == 33  # 1/3 = 33%
    assert "1 Done" in screen._lbl_kpi_summary.text()
    assert "1 Pending" in screen._lbl_kpi_summary.text()
    assert "1 Failed" in screen._lbl_kpi_summary.text()


def test_workspace_job_signals_handling(qtbot) -> None:
    """Verifies that progress, status, and completion signals from JobManager update UI rows."""
    mock_bridge = MagicMock()
    mock_jm = MagicMock()

    screen = WorkspaceScreen(bridge=mock_bridge, job_manager=mock_jm)
    qtbot.addWidget(screen)

    project = Project(id=1, name="Test Series", source_folder="/m", output_folder="/o")
    ep = Episode(
        id=50,
        project_id=1,
        episode_number=1,
        source_file="/m/ep1.mp4",
        status=EpisodeStatus.PENDING,
    )
    project.episodes = [ep]

    screen._on_project_loaded(project)

    # 1. Progress signal
    screen._on_job_progress_updated(50, "Speech-to-Text", 45)
    assert screen._row_progress_bars[50].value() == 45
    assert "45%" in screen._row_stage_labels[50].text()

    # 2. Status change signal
    screen._on_job_status_changed(50, EpisodeStatus.TRANSCRIBING.value)
    assert "Transcribing" in screen._row_status_labels[50].text()

    # 3. Episode completed signal
    screen._on_job_episode_completed(50)
    assert screen._row_progress_bars[50].value() == 100
    assert "Done" in screen._row_status_labels[50].text()
    assert screen._row_action_buttons[50].text() == "Review"


def test_workspace_controls_and_back_signal(qtbot) -> None:
    """Verifies that Start All, Pause All, and back button interact with JobManager and signals."""
    mock_bridge = MagicMock()
    mock_jm = MagicMock()

    screen = WorkspaceScreen(bridge=mock_bridge, job_manager=mock_jm)
    qtbot.addWidget(screen)

    project = Project(id=1, name="Test Series", source_folder="/m", output_folder="/o")
    ep1 = Episode(
        id=1, project_id=1, episode_number=1, source_file="/m/ep1.mp4", status=EpisodeStatus.PENDING
    )
    project.episodes = [ep1]
    screen._on_project_loaded(project)

    # Start all
    screen._btn_start_all.click()
    mock_jm.enqueue.assert_called_with(1)

    # Pause all
    screen._btn_pause_all.click()
    mock_jm.stop.assert_called_once()

    # Back to projects
    with qtbot.waitSignal(screen.back_requested, timeout=1000):
        screen._btn_back.click()
