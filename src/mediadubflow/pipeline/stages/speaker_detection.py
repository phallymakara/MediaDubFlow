"""Stage 4 — Speaker diarization using pyannote.audio."""

from __future__ import annotations

from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class SpeakerDetectionStage(PipelineStage):
    """
    Identify and label individual speakers in the audio using pyannote.

    The resulting speaker segments are saved as RTTM and merged with the
    transcript so each line is attributed to a specific speaker.
    """

    name = "Speaker Detection"

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Running speaker diarization", ctx.episode_id)
        # TODO: implement using pyannote.audio
        raise NotImplementedError("SpeakerDetectionStage is not yet implemented")

    def can_skip(self, ctx: StageContext) -> bool:
        return ctx.diarization_path is not None and ctx.diarization_path.exists()
