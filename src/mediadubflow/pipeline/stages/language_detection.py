"""Stage 1 — Detect the source language from the episode audio."""

from __future__ import annotations

import asyncio
from pathlib import Path

from faster_whisper import decode_audio
from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.whisper_model import get_whisper_model


def _detect_language_sync(audio_path: Path) -> tuple[str, float]:
    """Synchronous language detection helper executed in a background thread."""
    model = get_whisper_model()
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    language, probability, _ = model.detect_language(audio)
    return language, probability


class LanguageDetectionStage(PipelineStage):
    """
    Detect the spoken language using a short audio sample with Whisper.

    If confidence is below a configurable threshold the episode is
    paused for manual review instead of continuing automatically.
    """

    name = "Language Detection"

    # Minimum confidence score to proceed automatically.
    CONFIDENCE_THRESHOLD = 0.80

    def can_skip(self, ctx: StageContext) -> bool:
        """Skip if source language is already set (e.g. from user input or checkpoint)."""
        return bool(ctx.source_language)

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Starting language detection", ctx.episode_id)

        # Prefer extracted_audio if available, fallback to source video file
        if ctx.extracted_audio is not None and ctx.extracted_audio.exists():
            input_file = ctx.extracted_audio
        elif ctx.source_file.exists():
            input_file = ctx.source_file
        else:
            msg = (
                f"No valid audio or video file available for language detection: {ctx.source_file}"
            )
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        logger.debug("[Episode {}] Detecting language from {}", ctx.episode_id, input_file)

        try:
            language, confidence = await asyncio.to_thread(_detect_language_sync, input_file)
        except Exception as exc:
            msg = f"Language detection error: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.source_language = language
        ctx.language_confidence = confidence

        if confidence < self.CONFIDENCE_THRESHOLD:
            msg = (
                f"Detected language '{language}' with low confidence {confidence:.1%} "
                f"(threshold is {self.CONFIDENCE_THRESHOLD:.1%}). Manual review required."
            )
            logger.warning("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(
                success=False,
                message=msg,
                context_updates={
                    "source_language": language,
                    "language_confidence": confidence,
                    "needs_review": True,
                },
            )

        msg = f"Detected language '{language}' with confidence {confidence:.1%}"
        logger.info("[Episode {}] {}", ctx.episode_id, msg)
        return StageResult(
            success=True,
            message=msg,
            context_updates={
                "source_language": language,
                "language_confidence": confidence,
            },
        )
