"""
Unit tests for video combination logic and CombineVideosDialog.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mediadubflow.gui.dialogs.combine_videos_dialog import CombineVideosDialog
from mediadubflow.models.orm import Episode, EpisodeStatus, Project
from mediadubflow.utils.ffmpeg import concatenate_videos


@pytest.mark.asyncio
async def test_concatenate_videos_empty_list() -> None:
    """Verifies that concatenate_videos returns error if no paths provided."""
    code, stdout, stderr = await concatenate_videos([], Path("/tmp/out.mp4"))
    assert code != 0
    assert "No video files provided" in stderr


@pytest.mark.asyncio
async def test_concatenate_videos_single_file(tmp_path: Path) -> None:
    """Verifies that concatenate_videos copies single file directly."""
    src = tmp_path / "single.mp4"
    src.write_bytes(b"dummy video data")
    out = tmp_path / "out_single.mp4"

    code, stdout, stderr = await concatenate_videos([src], out)
    assert code == 0
    assert out.exists()
    assert out.read_bytes() == b"dummy video data"


@pytest.mark.asyncio
async def test_concatenate_videos_manifest_and_execution(tmp_path: Path, monkeypatch) -> None:
    """Verifies that concatenate_videos creates manifest and executes FFmpeg."""
    v1 = tmp_path / "ep1.mp4"
    v2 = tmp_path / "ep2.mp4"
    v1.write_bytes(b"ep1")
    v2.write_bytes(b"ep2")
    out = tmp_path / "combined.mp4"

    # Mock run_ffmpeg to simulate successful FFmpeg run
    async def mock_run_ffmpeg(args, timeout=600.0):
        # The output file argument is the last one in args
        target = Path(args[-1])
        target.write_bytes(b"mock combined video content")
        return 0, "FFmpeg success", ""

    monkeypatch.setattr("mediadubflow.utils.ffmpeg.run_ffmpeg", mock_run_ffmpeg)

    code, stdout, stderr = await concatenate_videos([v1, v2], out)
    assert code == 0
    assert out.exists()
    assert out.read_bytes() == b"mock combined video content"


def test_combine_videos_dialog_plan_generation(qtbot, tmp_path: Path) -> None:
    """Verifies batch slicing, filename formatting, and mode toggle in CombineVideosDialog."""
    mock_bridge = MagicMock()

    # Prepare project with 7 episodes, all with valid dubbed files
    project = Project(
        id=99999,
        name="The Royal Heir",
        source_folder=str(tmp_path / "sources"),
        output_folder=str(tmp_path / "output"),
    )

    episodes: list[Episode] = []
    for i in range(1, 8):
        v = tmp_path / f"ep_{i}_dubbed.mp4"
        v.write_bytes(b"video data")
        ep = Episode(
            id=i,
            project_id=99999,
            episode_number=i,
            title=f"The Royal Heir - Episode {i}",
            source_file=str(tmp_path / f"source_{i}.mp4"),
            output_video_path=str(v),
            status=EpisodeStatus.DONE,
        )
        episodes.append(ep)

    project.episodes = episodes

    # 1. Initialize dialog in batch mode with default batch_size=3
    dlg = CombineVideosDialog(
        project=project,
        episodes=episodes,
        bridge=mock_bridge,
        initial_mode="batch",
        initial_batch_size=3,
    )
    qtbot.addWidget(dlg)
    dlg._txt_movie_title.setText("The Royal Heir")

    # 7 episodes grouped into 3s -> 3 batches with sequential [ភាគ --] title of the movie
    assert len(dlg._batches) == 3
    assert dlg._batches[0].output_filename == "[ភាគ 01] The Royal Heir.mp4"
    assert dlg._batches[1].output_filename == "[ភាគ 02] The Royal Heir.mp4"
    assert dlg._batches[2].output_filename == "[ភាគ 03] The Royal Heir.mp4"

    # 2. Change batch size spinbox to 5 -> 2 batches: [ភាគ 01] and [ភាគ 02]
    dlg._spin_batch_size.setValue(5)
    assert len(dlg._batches) == 2
    assert dlg._batches[0].output_filename == "[ភាគ 01] The Royal Heir.mp4"
    assert dlg._batches[1].output_filename == "[ភាគ 02] The Royal Heir.mp4"

    # 3. Change starting ភាគ number to 10 -> [ភាគ 10] and [ភាគ 11]
    dlg._spin_start_part.setValue(10)
    assert dlg._batches[0].output_filename == "[ភាគ 10] The Royal Heir.mp4"
    assert dlg._batches[1].output_filename == "[ភាគ 11] The Royal Heir.mp4"

    # 4. Switch to Combine ALL mode -> 1 batch containing all 7 episodes as [ភាគ 10]
    dlg._rb_mode_all.setChecked(True)
    assert len(dlg._batches) == 1
    assert dlg._batches[0].output_filename == "[ភាគ 10] The Royal Heir.mp4"
    assert len(dlg._batches[0].episodes) == 7



def test_combine_videos_dialog_source_video_fallback(qtbot, tmp_path: Path) -> None:
    """Verifies that dialog falls back to source videos if no dubbed videos exist."""
    mock_bridge = MagicMock()

    project = Project(
        id=2,
        name="Legend of Hero",
        source_folder=str(tmp_path / "sources"),
        output_folder=str(tmp_path / "output"),
    )

    episodes: list[Episode] = []
    for i in range(1, 4):
        src = tmp_path / f"source_{i}.mp4"
        src.write_bytes(b"source data")
        ep = Episode(
            id=i,
            project_id=2,
            episode_number=i,
            source_file=str(src),
            output_video_path=None,  # Not dubbed yet
            status=EpisodeStatus.PENDING,
        )
        episodes.append(ep)

    project.episodes = episodes

    dlg = CombineVideosDialog(
        project=project,
        episodes=episodes,
        bridge=mock_bridge,
        initial_mode="batch",
        initial_batch_size=3,
    )
    qtbot.addWidget(dlg)

    # Dubbed radio button should be disabled because 0 dubbed files exist
    assert not dlg._rb_use_dubbed.isEnabled()
    # Source radio button should be checked
    assert dlg._rb_use_source.isChecked()
    # 1 batch containing the 3 source episodes
    assert len(dlg._batches) == 1
    assert len(dlg._batches[0].input_paths) == 3
