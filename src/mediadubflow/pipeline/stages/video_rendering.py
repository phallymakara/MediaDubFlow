"""Stage 9 — Render the final Khmer-localized video."""

from __future__ import annotations

from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class VideoRenderingStage(PipelineStage):
    """
    Combine the original video stream with the mixed Khmer audio and
    the generated subtitle track into the final output video.
    """

    name = "Video Rendering"

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Rendering final video", ctx.episode_id)
        # TODO: implement using ffmpeg-python
        raise NotImplementedError("VideoRenderingStage is not yet implemented")

    def can_skip(self, ctx: StageContext) -> bool:
        return ctx.output_video_path is not None and ctx.output_video_path.exists()
