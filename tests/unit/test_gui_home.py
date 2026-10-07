"""
Unit tests for HomeScreen, AsyncBridge, MediaDubFlowApp, and EditProjectDialog.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtWidgets import QPushButton

from mediadubflow.gui.dialogs.edit_project_dialog import EditProjectDialog
from mediadubflow.gui.screens.home_screen import HomeScreen
from mediadubflow.models.orm import Episode, EpisodeStatus, Project


def test_home_screen_empty_state(qtbot) -> None:
    """When no projects exist in database, empty state label is displayed."""
    mock_bridge = MagicMock()
    screen = HomeScreen(bridge=mock_bridge)
    qtbot.addWidget(screen)

    assert screen._btn_new.text() == "+ New Project"
    assert screen._search_input.placeholderText() == "Search recent projects..."
    assert screen._table.columnCount() == 5

    # Simulate zero projects loaded
    screen._on_projects_loaded([])
    assert not screen._table.isVisible()
    assert screen._empty_label.isVisible()
    assert "No recent projects found" in screen._empty_label.text()


def test_home_screen_new_project_signal(qtbot) -> None:
    """Clicking + New Project button emits new_project_requested signal."""
    mock_bridge = MagicMock()
    screen = HomeScreen(bridge=mock_bridge)
    qtbot.addWidget(screen)

    with qtbot.waitSignal(screen.new_project_requested, timeout=2000):
        screen._btn_new.click()


def test_home_screen_search_filtering(qtbot) -> None:
    """Searching filters the projects table in real time."""
    mock_bridge = MagicMock()
    screen = HomeScreen(bridge=mock_bridge)
    qtbot.addWidget(screen)

    p1 = Project(
        id=1,
        name="Royal Romance Drama",
        source_folder="/media/royal",
        output_folder="/output/royal",
        target_language="km",
    )
    p1.episodes = [Episode(status=EpisodeStatus.DONE)]

    p2 = Project(
        id=2,
        name="Action Thriller",
        source_folder="/media/action",
        output_folder="/output/action",
        target_language="km",
    )
    p2.episodes = []

    # Load projects
    screen._on_projects_loaded([p1, p2])
    assert screen._table.isVisible()
    assert screen._table.rowCount() == 2

    # Filter by 'royal'
    screen._search_input.setText("royal")
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Royal Romance Drama"

    # Filter with no matches
    screen._search_input.setText("comedy")
    assert not screen._table.isVisible()
    assert screen._empty_label.isVisible()
    assert "No projects matching 'comedy'" in screen._empty_label.text()

    # Clear filter
    screen._search_input.clear()
    assert screen._table.isVisible()
    assert screen._table.rowCount() == 2


def test_home_screen_table_actions_display(qtbot) -> None:
    """Project rows show episode count, status badge, created date, and CRUD action buttons."""
    mock_bridge = MagicMock()
    screen = HomeScreen(bridge=mock_bridge)
    qtbot.addWidget(screen)

    p = Project(
        id=10,
        name="Test Project",
        source_folder="/media/test",
        output_folder="/output/test",
        target_language="km",
    )
    p.episodes = [
        Episode(status=EpisodeStatus.DONE),
        Episode(status=EpisodeStatus.DONE),
    ]

    screen._on_projects_loaded([p])
    assert screen._table.rowCount() == 1

    # Col 0: Name
    assert screen._table.item(0, 0).text() == "Test Project"
    # Col 1: Episodes
    assert screen._table.item(0, 1).text() == "2 Episodes"
    # Col 2: Status widget exists
    status_cell = screen._table.cellWidget(0, 2)
    assert status_cell is not None
    # Col 4: Action button cell exists and contains the three-dot button
    action_cell = screen._table.cellWidget(0, 4)
    assert action_cell is not None
    button = action_cell.findChild(QPushButton)
    assert button is not None
    assert button.text() == "···"
    assert button.toolTip() == "Actions"



def test_home_screen_double_click_opens_project(qtbot) -> None:
    """Double-clicking a project row emits project_selected."""
    mock_bridge = MagicMock()
    screen = HomeScreen(bridge=mock_bridge)
    qtbot.addWidget(screen)

    p = Project(
        id=42,
        name="Double Click Me",
        source_folder="/media/dc",
        output_folder="/output/dc",
        target_language="km",
    )
    p.episodes = []
    screen._on_projects_loaded([p])

    with qtbot.waitSignal(screen.project_selected, timeout=2000) as blocker:
        screen._on_table_cell_double_clicked(0, 0)
    assert blocker.args == [42]


def test_edit_project_dialog_validation_and_values(qtbot) -> None:
    """EditProjectDialog validates name and returns updated values."""
    p = Project(
        id=1,
        name="Original Title",
        source_folder="/videos/orig",
        output_folder="/output/orig",
        target_language="km",
    )

    dlg = EditProjectDialog(project=p)
    qtbot.addWidget(dlg)

    assert dlg._txt_name.text() == "Original Title"
    assert dlg._txt_source.text() == "/videos/orig"
    assert dlg._txt_output.text() == "/output/orig"

    # Validation: empty name shows error and prevents accept
    dlg._txt_name.setText("")
    dlg._on_save_clicked()
    assert dlg._lbl_name_error.isVisible()
    assert "cannot be empty" in dlg._lbl_name_error.text()

    # Enter valid name
    dlg._txt_name.setText("Renamed Drama")
    dlg._txt_output.setText("/output/custom")
    dlg._on_save_clicked()

    updated = dlg.get_updated_values()
    assert updated["name"] == "Renamed Drama"
    assert updated["output_folder"] == "/output/custom"
    assert updated["target_language"] == "km"
