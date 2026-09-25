"""Stage 9 — Render the final Khmer-localized video."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.config.settings import PipelineOutputMode
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.ffmpeg import run_ffmpeg


class VideoRenderingStage(PipelineStage):
    """
    Combine the original video stream with the mixed Khmer dubbed audio
    into the final output video (.mp4).
    """

    name = "Video Rendering"

    def can_skip(self, ctx: StageContext) -> bool:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            return True
        return (
            ctx.output_video_path is not None
            and ctx.output_video_path.exists()
            and ctx.output_video_path.stat().st_size > 0
        )

    async def run(self, ctx: StageContext) -> StageResult:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            logger.info(
                "[Episode {}] Skipping video rendering (output_mode=subtitles_only)",
                ctx.episode_id,
            )
            return StageResult(success=True, message="Video rendering skipped per output mode setting")

        logger.info("[Episode {}] Rendering final dubbed video", ctx.episode_id)

        # Output video goes to the final output directory
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        stem = ctx.source_file.stem
        output_video = ctx.output_dir / f"{stem}_dubbed.mp4"

        # Determine audio source (prefer mixed audio, fallback to tts audio)
        audio_source = ctx.mixed_audio_path or ctx.tts_audio_path
        if audio_source is None or not audio_source.exists():
            msg = f"No dubbed audio track found to render video for episode {ctx.episode_id}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        logger.info(
            "[Episode {}] Muxing video '{}' with audio '{}' -> '{}'",
            ctx.episode_id,
            ctx.source_file.name,
            audio_source.name,
            output_video.name,
        )

        # Fast muxing: copy video stream directly (-c:v copy), encode audio to AAC
        args = [
            "-y",
            "-i",
            str(ctx.source_file),
            "-i",
            str(audio_source),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_video),
        ]

        code, _, stderr = await run_ffmpeg(args)
        if code != 0:
            msg = f"Video rendering failed: {stderr[:200]}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.output_video_path = output_video
        logger.info("[Episode {}] Final dubbed video rendered at {}", ctx.episode_id, output_video)
        return StageResult(
            success=True,
            message="Final dubbed video rendered successfully",
            context_updates={"output_video_path": output_video},
        )
