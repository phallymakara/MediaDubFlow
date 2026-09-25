"""
Unit tests for TranscriptionStage and JSON serialization of Faster-Whisper segments.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.transcription import (
    TranscriptionStage,
    _transcribe_audio_sync,
)


def test_can_skip_when_transcript_exists(tmp_path: Path) -> None:
    """can_skip returns True when transcript_path exists and is non-empty."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        transcript_path=transcript,
    )
    stage = TranscriptionStage()
    assert stage.can_skip(ctx) is True


def test_can_skip_when_transcript_missing_or_none(tmp_path: Path) -> None:
    """can_skip returns False when transcript_path is None or does not exist."""
    stage = TranscriptionStage()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        transcript_path=None,
    )
    assert stage.can_skip(ctx) is False

    ctx.transcript_path = tmp_path / "missing.json"
    assert stage.can_skip(ctx) is False


def test_can_skip_when_transcript_empty(tmp_path: Path) -> None:
    """can_skip returns False when transcript_path exists but is 0 bytes."""
    empty_file = tmp_path / "empty.json"
    empty_file.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        transcript_path=empty_file,
    )
    stage = TranscriptionStage()
    assert stage.can_skip(ctx) is False


@pytest.mark.asyncio
async def test_transcription_missing_input_audio(tmp_path: Path) -> None:
    """run returns StageResult(success=False) when no audio/video file exists."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "missing_video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=None,
    )
    stage = TranscriptionStage()
    result = await stage.run(ctx)

    assert result.success is False
    assert "No valid audio or video file found" in result.message


@pytest.mark.asyncio
async def test_transcription_prefers_extracted_audio(tmp_path: Path) -> None:
    """Transcription prefers extracted_audio over source_file."""
    video = tmp_path / "video.mp4"
    video.touch()
    audio = tmp_path / "audio.wav"
    audio.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=audio,
    )

    stage = TranscriptionStage()
    with patch(
        "mediadubflow.pipeline.stages.transcription._transcribe_audio_sync",
        return_value=(5, "zh", 60.0),
    ) as mock_sync:
        expected_json = tmp_path / "transcript_1.json"
        expected_json.write_text("[]", encoding="utf-8")

        result = await stage.run(ctx)
        assert result.success is True
        mock_sync.assert_called_once_with(audio, expected_json, "")


@pytest.mark.asyncio
async def test_transcription_fallback_to_source_file(tmp_path: Path) -> None:
    """Transcription falls back to source_file when extracted_audio is None."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=2,
        project_id=1,
        source_file=video,
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=None,
    )

    stage = TranscriptionStage()
    with patch(
        "mediadubflow.pipeline.stages.transcription._transcribe_audio_sync",
        return_value=(3, "en", 30.0),
    ) as mock_sync:
        expected_json = tmp_path / "transcript_2.json"
        expected_json.write_text("[]", encoding="utf-8")

        result = await stage.run(ctx)
        assert result.success is True
        mock_sync.assert_called_once_with(video, expected_json, "")


@pytest.mark.asyncio
async def test_transcription_exception_handling(tmp_path: Path) -> None:
    """run catches unexpected exceptions and returns failure."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path,
        output_dir=tmp_path,
    )

    stage = TranscriptionStage()
    with patch(
        "mediadubflow.pipeline.stages.transcription._transcribe_audio_sync",
        side_effect=RuntimeError("Whisper decode failed"),
    ):
        result = await stage.run(ctx)
        assert result.success is False
        assert "Transcription error" in result.message


def test_transcribe_audio_sync_helper(tmp_path: Path) -> None:
    """_transcribe_audio_sync invokes WhisperModel and persists serialized JSON."""
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    output_json = tmp_path / "transcript.json"

    # Mock segment with word timestamps
    mock_word_1 = SimpleNamespace(word="Hello", start=0.5, end=0.9, probability=0.99)
    mock_word_2 = SimpleNamespace(word="world", start=1.0, end=1.5, probability=0.96)
    mock_segment = SimpleNamespace(
        id=0,
        start=0.5,
        end=1.5,
        text="Hello world",
        words=[mock_word_1, mock_word_2],
    )

    mock_info = SimpleNamespace(language="zh", duration=45.2)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch(
        "mediadubflow.pipeline.stages.transcription.get_whisper_model", return_value=mock_model
    ):
        count, lang, duration = _transcribe_audio_sync(
            audio_path, output_json, source_language="zh"
        )

        assert count == 1
        assert lang == "zh"
        assert duration == 45.2
        assert output_json.exists()

        mock_model.transcribe.assert_called_once_with(
            str(audio_path),
            language="zh",
            word_timestamps=True,
            vad_filter=True,
        )

        with output_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["text"] == "Hello world"
        assert data[0]["start"] == 0.5
        assert data[0]["end"] == 1.5
        assert len(data[0]["words"]) == 2
        assert data[0]["words"][0]["word"] == "Hello"
