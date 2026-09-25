"""Stage 7 — Generate Khmer speech audio using TTS."""

from __future__ import annotations

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class TTSGenerationStage(PipelineStage):
    """
    Synthesize Khmer speech for each translated segment.

    Each speaker is mapped to a consistent TTS voice so the same
    character always sounds the same within and across episodes.
    """

    name = "TTS Generation"

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info(
            "[Episode {}] Generating Khmer TTS audio (backend={})",
            ctx.episode_id,
            settings.tts_backend,
        )
        # TODO: implement using Coqui TTS or OpenAI TTS API
        raise NotImplementedError("TTSGenerationStage is not yet implemented")

    def can_skip(self, ctx: StageContext) -> bool:
        return ctx.tts_audio_path is not None and ctx.tts_audio_path.exists()
