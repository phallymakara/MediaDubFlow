"""Stage 8 — Mix Khmer dialogue TTS with background music/SFX."""

from __future__ import annotations

from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class AudioMixingStage(PipelineStage):
    """
    Blend synthesized Khmer dialogue with the separated background
    audio track (music and sound effects).

    Applies LUFS loudness normalization so the dubbed dialogue sits at
    the same perceived volume as the original soundtrack.
    """

    name = "Audio Mixing"

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Mixing Khmer dialogue with background audio", ctx.episode_id)
        # TODO: implement using ffmpeg-python
        raise NotImplementedError("AudioMixingStage is not yet implemented")

    def can_skip(self, ctx: StageContext) -> bool:
        return ctx.mixed_audio_path is not None and ctx.mixed_audio_path.exists()
