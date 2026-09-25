"""Basic smoke tests — verify the package imports and configuration loads."""

from __future__ import annotations

from pathlib import Path

import pytest

from mediadubflow.utils.episode_detector import detect_episodes


def test_settings_loads() -> None:
    """Settings should load without error using defaults."""
    from mediadubflow.config import settings

    assert settings.database_url is not None


def test_episode_detector_missing_folder() -> None:
    """detect_episodes raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        detect_episodes(Path("/non/existent/path"))


def test_episode_number_extraction(tmp_path: Path) -> None:
    """detect_episodes correctly extracts episode numbers from filenames."""
    from mediadubflow.utils.episode_detector import detect_episodes

    # Create dummy video files.
    files = ["Drama_E01.mp4", "Drama_E02.mp4", "Drama_E10.mp4"]
    for name in files:
        (tmp_path / name).touch()

    results = detect_episodes(tmp_path)
    assert [n for n, _ in results] == [1, 2, 10]
