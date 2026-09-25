"""
Unit tests for FFmpeg execution utilities and the AudioExtractionStage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.audio_extraction import AudioExtractionStage
from mediadubflow.utils.ffmpeg import (
    extract_audio_track,
    get_ffmpeg_path,
    run_ffmpeg,
)


def test_get_ffmpeg_path_found() -> None:
    """get_ffmpeg_path returns the binary path when found on PATH."""
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        assert get_ffmpeg_path() == "/usr/bin/ffmpeg"


def test_get_ffmpeg_path_not_found() -> None:
    """get_ffmpeg_path raises FileNotFoundError when ffmpeg is missing."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="ffmpeg executable was not found"):
            get_ffmpeg_path()


@pytest.mark.asyncio
async def test_run_ffmpeg_success() -> None:
    """run_ffmpeg returns returncode 0, stdout, and stderr on success."""
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"stdout log", b"stderr log"))
    mock_proc.returncode = 0

    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
    ):
        code, stdout, stderr = await run_ffmpeg(["-version"])

        assert code == 0
        assert stdout == "stdout log"
        assert stderr == "stderr log"
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_run_ffmpeg_failure() -> None:
    """run_ffmpeg returns non-zero returncode on failure."""
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"Conversion failed"))
    mock_proc.returncode = 1

    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        code, _, stderr = await run_ffmpeg(["-i", "bad.mp4"])
        assert code == 1
        assert "Conversion failed" in stderr


@pytest.mark.asyncio
async def test_extract_audio_track_parameters(tmp_path: Path) -> None:
    """extract_audio_track passes correct 16kHz mono PCM arguments to run_ffmpeg."""
    source = tmp_path / "video.mp4"
    dest = tmp_path / "audio.wav"

    with patch("mediadubflow.utils.ffmpeg.run_ffmpeg", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (0, "", "")
        await extract_audio_track(source, dest, sample_rate=16000, channels=1)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "-vn" in args
        assert "-acodec" in args and "pcm_s16le" in args
        assert "-ar" in args and "16000" in args
        assert "-ac" in args and "1" in args
        assert str(source) in args
        assert str(dest) in args


def test_can_skip_when_file_exists(tmp_path: Path) -> None:
    """can_skip returns True when extracted_audio exists and is non-empty."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=audio_file,
    )

    stage = AudioExtractionStage()
    assert stage.can_skip(ctx) is True


def test_can_skip_when_file_missing(tmp_path: Path) -> None:
    """can_skip returns False when extracted_audio is None or does not exist."""
    stage = AudioExtractionStage()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=None,
    )
    assert stage.can_skip(ctx) is False

    ctx.extracted_audio = tmp_path / "nonexistent.wav"
    assert stage.can_skip(ctx) is False


def test_can_skip_when_file_empty(tmp_path: Path) -> None:
    """can_skip returns False when extracted_audio exists but is 0 bytes."""
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=empty_file,
    )

    stage = AudioExtractionStage()
    assert stage.can_skip(ctx) is False


@pytest.mark.asyncio
async def test_audio_extraction_missing_source(tmp_path: Path) -> None:
    """run returns StageResult(success=False) when source video is missing."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "missing.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
    )

    stage = AudioExtractionStage()
    result = await stage.run(ctx)

    assert result.success is False
    assert "Source file does not exist" in result.message


@pytest.mark.asyncio
async def test_audio_extraction_ffmpeg_not_found(tmp_path: Path) -> None:
    """run returns StageResult(success=False) when ffmpeg binary is not found."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )

    stage = AudioExtractionStage()
    with patch(
        "mediadubflow.pipeline.stages.audio_extraction.extract_audio_track",
        side_effect=FileNotFoundError("ffmpeg not found"),
    ):
        result = await stage.run(ctx)
        assert result.success is False
        assert "ffmpeg not found" in result.message


@pytest.mark.asyncio
async def test_audio_extraction_ffmpeg_failure(tmp_path: Path) -> None:
    """run returns StageResult(success=False) when ffmpeg exits with error."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )

    stage = AudioExtractionStage()
    with patch(
        "mediadubflow.pipeline.stages.audio_extraction.extract_audio_track",
        new_callable=AsyncMock,
    ) as mock_extract:
        mock_extract.return_value = (1, "", "Invalid data found when processing input")
        result = await stage.run(ctx)

        assert result.success is False
        assert "FFmpeg audio extraction failed" in result.message
        assert "Invalid data" in result.message


@pytest.mark.asyncio
async def test_audio_extraction_success(tmp_path: Path) -> None:
    """run successfully extracts audio, sets extracted_audio in ctx and returns success."""
    video = tmp_path / "video.mp4"
    video.touch()

    work_dir = tmp_path / "work"

    ctx = StageContext(
        episode_id=42,
        project_id=1,
        source_file=video,
        work_dir=work_dir,
        output_dir=tmp_path / "out",
    )

    expected_wav = work_dir / "extracted_audio_42.wav"

    async def fake_extract(source_file, output_wav, **kwargs):
        # Simulate FFmpeg creating the file
        output_wav.write_bytes(b"RIFF" + b"\x00" * 200)
        return 0, "mock output", ""

    stage = AudioExtractionStage()
    with patch(
        "mediadubflow.pipeline.stages.audio_extraction.extract_audio_track",
        side_effect=fake_extract,
    ):
        result = await stage.run(ctx)

        assert result.success is True
        assert ctx.extracted_audio == expected_wav
        assert result.context_updates.get("extracted_audio") == expected_wav
        assert expected_wav.exists()


@pytest.mark.asyncio
async def test_audio_extraction_insufficient_disk_space(tmp_path: Path) -> None:
    """AudioExtractionStage aborts with an informative error if free disk space is under 500 MB."""
    from collections import namedtuple
    from unittest.mock import patch

    Usage = namedtuple("Usage", ["total", "used", "free"])
    mock_usage = Usage(total=100_000_000, used=90_000_000, free=10_000_000)  # ~10 MB free

    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )

    stage = AudioExtractionStage()
    with patch("shutil.disk_usage", return_value=mock_usage):
        result = await stage.run(ctx)
        assert result.success is False
        assert "Insufficient disk space" in result.message


@pytest.mark.asyncio
async def test_audio_extraction_ffmpeg_timeout(tmp_path: Path) -> None:
    """AudioExtractionStage handles TimeoutError from extract_audio_track gracefully."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )

    stage = AudioExtractionStage()
    with patch(
        "mediadubflow.pipeline.stages.audio_extraction.extract_audio_track",
        side_effect=TimeoutError("FFmpeg command timed out after 600 seconds"),
    ):
        result = await stage.run(ctx)
        assert result.success is False
        assert "timed out" in result.message
