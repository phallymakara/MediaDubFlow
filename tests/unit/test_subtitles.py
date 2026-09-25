"""
Unit tests for SRT / ASS subtitle generation and SubtitleGenerationStage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.subtitle_generation import SubtitleGenerationStage
from mediadubflow.utils.subtitles import create_ass_file, create_srt_file


def test_create_srt_file_content(tmp_path: Path) -> None:
    """create_srt_file writes a valid SubRip file with accurate timecodes."""
    out_srt = tmp_path / "test.srt"
    segments = [
        {"id": 0, "start": 1.5, "end": 4.25, "translated_text": "សួស្តី"},
        {"id": 1, "start": 5.0, "end": 8.0, "translated_text": "សុខសប្បាយជាទេ?"},
    ]

    result_path = create_srt_file(segments, out_srt)
    assert result_path == out_srt
    assert out_srt.exists()

    content = out_srt.read_text(encoding="utf-8")
    assert "1\n00:00:01,500 --> 00:00:04,250\nសួស្តី" in content
    assert "2\n00:00:05,000 --> 00:00:08,000\nសុខសប្បាយជាទេ?" in content


def test_create_ass_file_content(tmp_path: Path) -> None:
    """create_ass_file writes a valid ASS file with 1080p resolution and zero shadow."""
    out_ass = tmp_path / "test.ass"
    segments = [
        {"id": 0, "start": 1.0, "end": 3.0, "translated_text": "ជំរាបសួរ"},
    ]

    result_path = create_ass_file(segments, out_ass, font_name="Kantumruy Pro")
    assert result_path == out_ass
    assert out_ass.exists()

    content = out_ass.read_text(encoding="utf-8-sig")
    assert "[Script Info]" in content
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content
    assert "Kantumruy Pro" in content
    assert "Shadow=0" in content or ",0," in content  # Shadow is disabled
    assert "Dialogue: 0,0:00:01.00,0:00:03.00,KhmerDefault" in content
    assert "ជំរាបសួរ" in content


def test_can_skip_when_both_files_exist(tmp_path: Path) -> None:
    """can_skip returns True only when both .srt and .ass files exist and are non-empty."""
    srt_file = tmp_path / "sub.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi", encoding="utf-8")
    ass_file = tmp_path / "sub.ass"
    ass_file.write_text("[Script Info]", encoding="utf-8")

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        subtitle_srt_path=srt_file,
        subtitle_ass_path=ass_file,
    )

    stage = SubtitleGenerationStage()
    assert stage.can_skip(ctx) is True


def test_can_skip_when_missing_or_empty(tmp_path: Path) -> None:
    """can_skip returns False if either file is missing, None, or 0 bytes."""
    stage = SubtitleGenerationStage()

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        subtitle_srt_path=None,
        subtitle_ass_path=None,
    )
    assert stage.can_skip(ctx) is False

    srt_file = tmp_path / "sub.srt"
    srt_file.write_text("content", encoding="utf-8")
    ctx.subtitle_srt_path = srt_file
    assert stage.can_skip(ctx) is False

    empty_ass = tmp_path / "empty.ass"
    empty_ass.touch()
    ctx.subtitle_ass_path = empty_ass
    assert stage.can_skip(ctx) is False


@pytest.mark.asyncio
async def test_subtitle_stage_missing_input(tmp_path: Path) -> None:
    """run returns StageResult(success=False) when no translation/transcript exists."""
    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        translation_path=None,
        transcript_path=None,
    )

    stage = SubtitleGenerationStage()
    result = await stage.run(ctx)
    assert result.success is False
    assert "No translation or transcript file available" in result.message


@pytest.mark.asyncio
async def test_subtitle_stage_prefers_translation_path(tmp_path: Path) -> None:
    """Stage uses translation_path when available."""
    trans_file = tmp_path / "trans.json"
    trans_file.write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 2.0, "translated_text": "ខ្មែរ"}]),
        encoding="utf-8",
    )

    ctx = StageContext(
        episode_id=10,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        translation_path=trans_file,
    )

    stage = SubtitleGenerationStage()
    result = await stage.run(ctx)

    assert result.success is True
    assert ctx.subtitle_srt_path == tmp_path / "out" / "episode_10.srt"
    assert ctx.subtitle_ass_path == tmp_path / "out" / "episode_10.ass"
    assert ctx.subtitle_srt_path.exists()
    assert ctx.subtitle_ass_path.exists()
    assert "ខ្មែរ" in ctx.subtitle_srt_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_subtitle_stage_fallback_to_transcript_path(tmp_path: Path) -> None:
    """Stage falls back to transcript_path when translation_path is None."""
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps([{"id": 0, "start": 1.0, "end": 3.0, "text": "Original speech"}]),
        encoding="utf-8",
    )

    ctx = StageContext(
        episode_id=20,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        translation_path=None,
        transcript_path=transcript,
    )

    stage = SubtitleGenerationStage()
    result = await stage.run(ctx)

    assert result.success is True
    assert "Original speech" in ctx.subtitle_srt_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_subtitle_stage_empty_segments(tmp_path: Path) -> None:
    """Stage handles empty segment list gracefully."""
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]", encoding="utf-8")

    ctx = StageContext(
        episode_id=30,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        translation_path=empty_json,
    )

    stage = SubtitleGenerationStage()
    result = await stage.run(ctx)

    assert result.success is True
    assert ctx.subtitle_srt_path.exists()
    assert ctx.subtitle_ass_path.exists()


@pytest.mark.asyncio
async def test_subtitle_stage_invalid_json(tmp_path: Path) -> None:
    """Stage returns failure when JSON cannot be parsed."""
    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text("corrupted content", encoding="utf-8")

    ctx = StageContext(
        episode_id=40,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        translation_path=invalid_json,
    )

    stage = SubtitleGenerationStage()
    result = await stage.run(ctx)
    assert result.success is False
    assert "Failed to read segments" in result.message
