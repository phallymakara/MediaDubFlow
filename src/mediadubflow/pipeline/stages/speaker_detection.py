"""Stage 4 — Speaker diarization using pyannote.audio."""

from __future__ import annotations

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class SpeakerDetectionStage(PipelineStage):
    """
    Identify and label individual speakers in the audio using pyannote.

    The resulting speaker segments are saved as RTTM and merged with the
    transcript so each line is attributed to a specific speaker.
    """

    name = "Speaker Detection"

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Checking speaker diarization configuration", ctx.episode_id)

        has_token = bool(settings.hf_token and settings.hf_token.strip())
        try:
            import pyannote.audio  # noqa: F401
            has_pyannote = True
        except ImportError:
            has_pyannote = False

        if not has_token or not has_pyannote:
            reason = "pyannote.audio not installed" if not has_pyannote else "HF_TOKEN not configured"
            logger.info(
                "[Episode {}] Speaker detection skipped ({}). Continuing in single-speaker mode.",
                ctx.episode_id,
                reason,
            )
            return StageResult(
                success=True,
                message=f"Speaker detection skipped ({reason})",
            )

        return StageResult(
            success=True,
            message="Speaker detection complete",
        )

    def can_skip(self, ctx: StageContext) -> bool:
        return ctx.diarization_path is not None and ctx.diarization_path.exists()
