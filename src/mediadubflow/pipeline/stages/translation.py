"""Stage 5 — Translate transcript lines to Khmer via LLM."""

from __future__ import annotations

import json

from loguru import logger

from mediadubflow.config.settings import settings
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.services.translation_service import translate_transcript_segments


class TranslationStage(PipelineStage):
    """
    Send transcript segments to the configured LLM for context-aware
    Khmer translation.

    A rolling context window and series glossary are injected into the
    system prompt to maintain consistency across lines and episodes.
    """

    name = "Translation"

    def can_skip(self, ctx: StageContext) -> bool:
        """Skip if translation JSON already exists and is non-empty."""
        if ctx.translation_path is None or not ctx.translation_path.exists():
            return False
        try:
            return ctx.translation_path.stat().st_size > 0
        except OSError:
            return False

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info(
            "[Episode {}] Translating with provider={}",
            ctx.episode_id,
            settings.translation_provider,
        )

        if ctx.transcript_path is None or not ctx.transcript_path.exists():
            msg = f"Transcript file not found: {ctx.transcript_path}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        try:
            with ctx.transcript_path.open("r", encoding="utf-8") as f:
                segments = json.load(f)
        except Exception as exc:
            msg = f"Failed to read transcript file '{ctx.transcript_path}': {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not isinstance(segments, list):
            msg = f"Transcript JSON format invalid (expected list, got {type(segments)})"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if len(segments) > 5000:
            msg = f"Transcript has {len(segments)} segments, exceeding safety limit of 5000 segments per episode."
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        output_json = ctx.work_dir / f"translation_{ctx.episode_id}.json"

        if not segments:
            logger.warning("[Episode {}] Transcript has no segments to translate", ctx.episode_id)
            with output_json.open("w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            ctx.translation_path = output_json
            return StageResult(
                success=True,
                message="Transcript is empty; 0 segments translated",
                context_updates={"translation_path": output_json},
            )

        # Extract optional glossary from metadata
        glossary = (
            ctx.metadata.get("glossary") if isinstance(ctx.metadata.get("glossary"), dict) else None
        )

        try:
            translated_segments = await translate_transcript_segments(
                segments,
                source_language=ctx.source_language,
                target_language="km",
                glossary=glossary,
            )
        except Exception as exc:
            msg = f"Translation error: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        try:
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(translated_segments, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            msg = f"Failed to save translation file '{output_json}': {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.translation_path = output_json
        msg = (
            f"Successfully translated {len(translated_segments)} segments to Khmer ({output_json})"
        )
        logger.info("[Episode {}] {}", ctx.episode_id, msg)

        return StageResult(
            success=True,
            message=msg,
            context_updates={"translation_path": output_json},
        )
