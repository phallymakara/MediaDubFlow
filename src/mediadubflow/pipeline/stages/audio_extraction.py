"""Stage 2 — Extract audio track from the episode video using FFmpeg."""

from __future__ import annotations

import shutil

from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.ffmpeg import extract_audio_track


class AudioExtractionStage(PipelineStage):
    """
    Extract a 16 kHz mono WAV audio stream from the source video.

    The extracted audio is stored in the episode work directory and used
    by subsequent stages (transcription, diarization, TTS mixing).
    """

    name = "Audio Extraction"

    def can_skip(self, ctx: StageContext) -> bool:
        """Return True if a non-empty extracted audio file already exists."""
        if ctx.extracted_audio is None or not ctx.extracted_audio.exists():
            return False
        try:
            return ctx.extracted_audio.stat().st_size > 0
        except OSError:
            return False

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Extracting audio from {}", ctx.episode_id, ctx.source_file)

        if not ctx.source_file.exists():
            msg = f"Source file does not exist: {ctx.source_file}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)

        # Pre-flight check: Verify sufficient disk space for uncompressed WAV
        try:
            free_bytes = shutil.disk_usage(ctx.work_dir).free
            if free_bytes < 500 * 1024 * 1024:  # 500 MB minimum
                msg = (
                    f"Insufficient disk space: {free_bytes / (1024 * 1024):.1f} MB "
                    "remaining. Minimum 500 MB required."
                )
                logger.error("[Episode {}] {}", ctx.episode_id, msg)
                return StageResult(success=False, message=msg)
        except OSError as disk_err:
            logger.warning("[Episode {}] Could not check disk usage: {}", ctx.episode_id, disk_err)

        output_wav = ctx.work_dir / f"extracted_audio_{ctx.episode_id}.wav"

        try:
            returncode, _stdout, stderr = await extract_audio_track(
                source_file=ctx.source_file,
                output_wav=output_wav,
                sample_rate=16000,
                channels=1,
            )
        except FileNotFoundError as exc:
            logger.error("[Episode {}] FFmpeg binary not found: {}", ctx.episode_id, exc)
            return StageResult(success=False, message=str(exc))
        except Exception as exc:
            logger.error(
                "[Episode {}] Unexpected error during audio extraction: {}", ctx.episode_id, exc
            )
            return StageResult(success=False, message=f"Audio extraction error: {exc}")

        if returncode != 0:
            err_snippet = (
                stderr.strip().splitlines()[-1]
                if stderr.strip()
                else f"Process exited with code {returncode}"
            )
            msg = f"FFmpeg audio extraction failed: {err_snippet}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not output_wav.exists() or output_wav.stat().st_size == 0:
            msg = f"Audio extraction produced empty or missing file: {output_wav}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.extracted_audio = output_wav
        logger.info("[Episode {}] Audio successfully extracted to {}", ctx.episode_id, output_wav)
        return StageResult(
            success=True,
            message=f"Audio extracted successfully to {output_wav}",
            context_updates={"extracted_audio": output_wav},
        )
