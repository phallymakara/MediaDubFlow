"""
Unit tests for the LanguageDetectionStage and WhisperModel caching manager.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.language_detection import (
    LanguageDetectionStage,
    _detect_language_sync,
)
from mediadubflow.utils.whisper_model import (
    clear_whisper_model_cache,
    get_whisper_model,
)


@pytest.fixture(autouse=True)
def _cleanup_whisper_cache():
    """Ensure cache is reset between tests."""
    clear_whisper_model_cache()
    yield
    clear_whisper_model_cache()


def test_can_skip_when_language_set(tmp_path: Path) -> None:
    """can_skip returns True when source_language is non-empty."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        source_language="zh",
    )
    stage = LanguageDetectionStage()
    assert stage.can_skip(ctx) is True


def test_can_skip_when_language_empty(tmp_path: Path) -> None:
    """can_skip returns False when source_language is empty."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        source_language="",
    )
    stage = LanguageDetectionStage()
    assert stage.can_skip(ctx) is False


@pytest.mark.asyncio
async def test_language_detection_missing_input_file(tmp_path: Path) -> None:
    """run returns failure when neither extracted audio nor source video exist."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "missing_video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=None,
    )
    stage = LanguageDetectionStage()
    result = await stage.run(ctx)

    assert result.success is False
    assert "No valid audio or video file available" in result.message


@pytest.mark.asyncio
async def test_language_detection_prefers_extracted_audio(tmp_path: Path) -> None:
    """Detection prefers extracted audio over the source video file."""
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

    stage = LanguageDetectionStage()
    with patch(
        "mediadubflow.pipeline.stages.language_detection._detect_language_sync",
        return_value=("zh", 0.95),
    ) as mock_detect:
        result = await stage.run(ctx)
        assert result.success is True
        mock_detect.assert_called_once_with(audio)
        assert ctx.source_language == "zh"
        assert ctx.language_confidence == 0.95


@pytest.mark.asyncio
async def test_language_detection_fallback_to_source_file(tmp_path: Path) -> None:
    """Detection falls back to source video when extracted audio is missing."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=None,
    )

    stage = LanguageDetectionStage()
    with patch(
        "mediadubflow.pipeline.stages.language_detection._detect_language_sync",
        return_value=("ko", 0.92),
    ) as mock_detect:
        result = await stage.run(ctx)
        assert result.success is True
        mock_detect.assert_called_once_with(video)
        assert ctx.source_language == "ko"
        assert ctx.language_confidence == 0.92


@pytest.mark.asyncio
async def test_language_detection_low_confidence_requires_review(tmp_path: Path) -> None:
    """When confidence is below 0.80, stage returns failure with needs_review flag."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path,
        output_dir=tmp_path,
    )

    stage = LanguageDetectionStage()
    with patch(
        "mediadubflow.pipeline.stages.language_detection._detect_language_sync",
        return_value=("ja", 0.65),
    ):
        result = await stage.run(ctx)
        assert result.success is False
        assert result.context_updates.get("needs_review") is True
        assert "low confidence" in result.message
        assert ctx.source_language == "ja"


@pytest.mark.asyncio
async def test_language_detection_exception_handling(tmp_path: Path) -> None:
    """run catches unexpected exceptions during detection and returns failure."""
    video = tmp_path / "video.mp4"
    video.touch()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=video,
        work_dir=tmp_path,
        output_dir=tmp_path,
    )

    stage = LanguageDetectionStage()
    with patch(
        "mediadubflow.pipeline.stages.language_detection._detect_language_sync",
        side_effect=RuntimeError("CUDA out of memory"),
    ):
        result = await stage.run(ctx)
        assert result.success is False
        assert "Language detection error" in result.message


def test_detect_language_sync_helper(tmp_path: Path) -> None:
    """_detect_language_sync decodes audio and delegates to WhisperModel."""
    audio_file = tmp_path / "sample.wav"
    audio_file.touch()

    mock_model = MagicMock()
    mock_model.detect_language.return_value = ("zh", 0.98, [("zh", 0.98), ("en", 0.02)])

    with (
        patch(
            "mediadubflow.pipeline.stages.language_detection.get_whisper_model",
            return_value=mock_model,
        ),
        patch(
            "mediadubflow.pipeline.stages.language_detection.decode_audio",
            return_value=b"mock_audio",
        ),
    ):
        lang, prob = _detect_language_sync(audio_file)
        assert lang == "zh"
        assert prob == 0.98


def test_whisper_model_manager_caching() -> None:
    """get_whisper_model caches loaded models by configuration."""
    with patch("mediadubflow.utils.whisper_model.WhisperModel") as mock_cls:
        mock_cls.side_effect = [MagicMock(), MagicMock()]
        instance_1 = get_whisper_model("base", "cpu", "int8")
        instance_2 = get_whisper_model("base", "cpu", "int8")

        assert instance_1 is instance_2
        mock_cls.assert_called_once_with(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8",
        )

        # Clear cache and verify new initialization
        clear_whisper_model_cache()
        instance_3 = get_whisper_model("base", "cpu", "int8")
        assert instance_3 is not instance_1
        assert mock_cls.call_count == 2
