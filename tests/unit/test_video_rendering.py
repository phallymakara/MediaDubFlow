"""
Unit tests for VideoRenderingStage.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mediadubflow.config.settings import PipelineOutputMode, settings
from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.video_rendering import VideoRenderingStage


@pytest.fixture
def base_context(tmp_path: Path) -> StageContext:
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_file = tmp_path / "The_Doctor's_Cursed_Bride_-_Ep_01.mp4"
    source_file.write_bytes(b"dummy video")

    mixed_audio = work_dir / "mixed_1.wav"
    mixed_audio.write_bytes(b"dummy audio")

    return StageContext(
        episode_id=1,
        project_id=10,
        source_file=source_file,
        work_dir=work_dir,
        output_dir=output_dir,
        mixed_audio_path=mixed_audio,
        metadata={
            "episode_number": 1,
            "title": "The Doctor's Cursed Bride",
            "khmer_title": "រឿង កូនក្រមុំអាថ៌កំបាំងរបស់វេជ្ជបណ្ឌិត - ភាគ ០១",
        },
    )


@pytest.mark.asyncio
async def test_video_rendering_produces_clean_khmer_title(base_context: StageContext) -> None:
    """Video rendering formats output as '[ភាគ 01] <Khmer Title>.mp4' in output_dir."""
    stage = VideoRenderingStage()

    async def mock_ffmpeg(args):
        # args[-1] is the temp output file in work_dir
        out_file = Path(args[-1])
        out_file.write_bytes(b"rendered video mp4")
        return 0, "", ""

    with patch("mediadubflow.pipeline.stages.video_rendering.run_ffmpeg", side_effect=mock_ffmpeg):
        result = await stage.run(base_context)

    assert result.success is True
    assert base_context.output_video_path is not None
    assert base_context.output_video_path.exists()
    assert base_context.output_video_path.parent == base_context.output_dir

    # Name must have [ភាគ 01] and the Khmer title, without duplicate episode suffixes
    assert base_context.output_video_path.name == "[ភាគ 01] រឿង កូនក្រមុំអាថ៌កំបាំងរបស់វេជ្ជបណ្ឌិត.mp4"


@pytest.mark.asyncio
async def test_video_rendering_cleans_stale_files_from_output_dir(base_context: StageContext) -> None:
    """Stale duplicate video files and series_glossary.json are removed from output_dir."""
    # Create stale files in output_dir
    stale_video = base_context.output_dir / "The_Doctor's_Cursed_Bride_-_Ep_01_dubbed.mp4"
    stale_video.write_bytes(b"old video")

    stale_glossary = base_context.output_dir / "series_glossary.json"
    stale_glossary.write_text(json.dumps({"test": "term"}))

    stage = VideoRenderingStage()

    async def mock_ffmpeg(args):
        out_file = Path(args[-1])
        out_file.write_bytes(b"new rendered video")
        return 0, "", ""

    with patch("mediadubflow.pipeline.stages.video_rendering.run_ffmpeg", side_effect=mock_ffmpeg):
        result = await stage.run(base_context)

    assert result.success is True
    # Verify stale video and glossary are cleaned up from output_dir
    assert not stale_video.exists()
    assert not stale_glossary.exists()

    # Verify only the final dubbed video exists in output_dir
    files_in_output = list(base_context.output_dir.iterdir())
    assert len(files_in_output) == 1
    assert files_in_output[0] == base_context.output_video_path


@pytest.mark.asyncio
async def test_video_rendering_can_skip_when_properly_named(base_context: StageContext) -> None:
    """can_skip returns True only when file exists and is not an untranslated fallback."""
    stage = VideoRenderingStage()

    # When no video exists
    assert stage.can_skip(base_context) is False

    # When untranslated video exists but khmer_title is set
    untranslated = base_context.output_dir / "old_dubbed.mp4"
    untranslated.write_bytes(b"content")
    base_context.output_video_path = untranslated
    assert stage.can_skip(base_context) is False

    # When properly translated video exists
    translated = base_context.output_dir / "[ភាគ 01] រឿង កូនក្រមុំ.mp4"
    translated.write_bytes(b"content")
    base_context.output_video_path = translated
    assert stage.can_skip(base_context) is True


@pytest.mark.asyncio
async def test_video_rendering_consistent_series_title_across_episodes(tmp_path: Path) -> None:
    """Subsequent episodes reuse cached series title and increment episode number."""
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate cached series title for project 42
    p_cache = settings.cache_dir / "project_42"
    p_cache.mkdir(parents=True, exist_ok=True)
    (p_cache / "series_title_km.txt").write_text(
        "រឿង វិញ្ញាណប្រពន្ធអាពាហ៍ពិពាហ៍របស់វេជ្ជបណ្ឌិត",
        encoding="utf-8",
    )

    source_ep2 = tmp_path / "Doctor_Ep_02.mp4"
    source_ep2.write_bytes(b"dummy")
    mixed_audio = work_dir / "mixed_2.wav"
    mixed_audio.write_bytes(b"audio")

    ctx = StageContext(
        episode_id=2,
        project_id=42,
        source_file=source_ep2,
        work_dir=work_dir,
        output_dir=output_dir,
        mixed_audio_path=mixed_audio,
        metadata={"episode_number": 2},
    )

    stage = VideoRenderingStage()

    async def mock_ffmpeg(args):
        Path(args[-1]).write_bytes(b"rendered")
        return 0, "", ""

    with patch("mediadubflow.pipeline.stages.video_rendering.run_ffmpeg", side_effect=mock_ffmpeg):
        result = await stage.run(ctx)

    assert result.success is True
    assert ctx.output_video_path is not None
    assert ctx.output_video_path.name == "[ភាគ 02] រឿង វិញ្ញាណប្រពន្ធអាពាហ៍ពិពាហ៍របស់វេជ្ជបណ្ឌិត.mp4"

