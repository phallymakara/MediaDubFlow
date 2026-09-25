"""Stage 8 — Mix Khmer dialogue TTS with background music/SFX."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.config.settings import PipelineOutputMode
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.ffmpeg import run_ffmpeg


class AudioMixingStage(PipelineStage):
    """
    Blend synthesized Khmer dialogue with the original background audio track.

    Applies volume ducking so background audio sits at 30% volume while
    Khmer dialogue is delivered clearly.
    """

    name = "Audio Mixing"

    def can_skip(self, ctx: StageContext) -> bool:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            return True
        return (
            ctx.mixed_audio_path is not None
            and ctx.mixed_audio_path.exists()
            and ctx.mixed_audio_path.stat().st_size > 0
        )

    async def run(self, ctx: StageContext) -> StageResult:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            logger.info(
                "[Episode {}] Skipping audio mixing (output_mode=subtitles_only)",
                ctx.episode_id,
            )
            return StageResult(success=True, message="Audio mixing skipped per output mode setting")

        logger.info("[Episode {}] Mixing Khmer dialogue with background audio", ctx.episode_id)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        mixed_audio = ctx.work_dir / f"mixed_{ctx.episode_id}.wav"

        # If we have both original audio and synthesized TTS audio, blend them
        if ctx.tts_audio_path and ctx.tts_audio_path.exists():
            if ctx.extracted_audio and ctx.extracted_audio.exists():
                logger.info("[Episode {}] Blending background audio with voice dialogue", ctx.episode_id)
                # Filter: duck original audio to 0.3, voice dialogue at 1.2
                filter_complex = (
                    "[0:a]volume=0.3[bg];[1:a]volume=1.2[vox];"
                    "[bg][vox]amix=inputs=2:duration=first:dropout_transition=2[out]"
                )
                args = [
                    "-y",
                    "-i",
                    str(ctx.extracted_audio),
                    "-i",
                    str(ctx.tts_audio_path),
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[out]",
                    "-c:a",
                    "pcm_s16le",
                    str(mixed_audio),
                ]
                code, _, stderr = await run_ffmpeg(args)
                if code != 0:
                    msg = f"Audio mixing failed: {stderr[:200]}"
                    logger.error("[Episode {}] {}", ctx.episode_id, msg)
                    return StageResult(success=False, message=msg)
            else:
                # Use voice track directly
                mixed_audio = ctx.tts_audio_path
        elif ctx.extracted_audio and ctx.extracted_audio.exists():
            mixed_audio = ctx.extracted_audio
        else:
            msg = f"No audio track available to mix for episode {ctx.episode_id}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.mixed_audio_path = mixed_audio
        logger.info("[Episode {}] Audio mixing completed at {}", ctx.episode_id, mixed_audio)
        return StageResult(
            success=True,
            message="Audio mixed successfully",
            context_updates={"mixed_audio_path": mixed_audio},
        )
