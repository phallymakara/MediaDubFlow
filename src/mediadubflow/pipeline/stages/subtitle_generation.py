"""Stage 6 — Generate Khmer .SRT and .ASS subtitle files."""

from __future__ import annotations

import json

from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.subtitles import create_ass_file, create_srt_file


class SubtitleGenerationStage(PipelineStage):
    """
    Convert translated segments into standard subtitle formats.

    Produces both .srt (simple) and .ass (styled) files in the output
    directory. Timing is taken from the original Whisper timestamps.
    """

    name = "Subtitle Generation"

    def can_skip(self, ctx: StageContext) -> bool:
        """Return True if both .srt and .ass files already exist and are non-empty."""
        if (
            ctx.subtitle_srt_path is None
            or not ctx.subtitle_srt_path.exists()
            or ctx.subtitle_ass_path is None
            or not ctx.subtitle_ass_path.exists()
        ):
            return False
        try:
            return (
                ctx.subtitle_srt_path.stat().st_size > 0
                and ctx.subtitle_ass_path.stat().st_size > 0
            )
        except OSError:
            return False

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Generating Khmer subtitles", ctx.episode_id)

        # Prefer translation_path, fallback to transcript_path
        if ctx.translation_path is not None and ctx.translation_path.exists():
            input_json = ctx.translation_path
        elif ctx.transcript_path is not None and ctx.transcript_path.exists():
            input_json = ctx.transcript_path
        else:
            msg = f"No translation or transcript file available for episode {ctx.episode_id}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        try:
            with input_json.open("r", encoding="utf-8") as f:
                segments = json.load(f)
        except Exception as exc:
            msg = f"Failed to read segments from '{input_json}': {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not isinstance(segments, list):
            msg = f"Subtitle source data format invalid (expected list, got {type(segments)})"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        # Output subtitles into the final output directory
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        srt_path = ctx.output_dir / f"episode_{ctx.episode_id}.srt"
        ass_path = ctx.output_dir / f"episode_{ctx.episode_id}.ass"

        try:
            create_srt_file(segments, srt_path)
            create_ass_file(segments, ass_path)
        except Exception as exc:
            msg = f"Subtitle generation error: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.subtitle_srt_path = srt_path
        ctx.subtitle_ass_path = ass_path

        msg = f"Subtitles generated successfully: {srt_path.name}, {ass_path.name}"
        logger.info("[Episode {}] {}", ctx.episode_id, msg)

        return StageResult(
            success=True,
            message=msg,
            context_updates={
                "subtitle_srt_path": srt_path,
                "subtitle_ass_path": ass_path,
            },
        )
