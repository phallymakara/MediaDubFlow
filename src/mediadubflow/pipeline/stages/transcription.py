"""Stage 3 — Transcribe speech to text with timestamps using Whisper."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.whisper_model import get_whisper_model


def _transcribe_audio_sync(
    audio_path: Path,
    output_json_path: Path,
    source_language: str | None = None,
) -> tuple[int, str, float]:
    """
    Synchronous transcription helper to execute inside a background thread.

    Runs faster-whisper on the input audio, collects word-level timestamps,
    and writes the serialized segments to output_json_path.

    Returns:
        Tuple of (segment_count, detected_or_used_language, audio_duration).
    """
    model = get_whisper_model()

    # Pass language if specified, otherwise let Whisper auto-detect
    lang_param = source_language.strip() if (source_language and source_language.strip()) else None

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=lang_param,
        word_timestamps=True,
        vad_filter=True,
    )

    serialized_segments: list[dict[str, Any]] = []

    for seg in segments_gen:
        words_list: list[dict[str, Any]] = []
        if seg.words:
            for w in seg.words:
                words_list.append(
                    {
                        "word": w.word,
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "probability": round(float(w.probability), 3),
                    }
                )

        serialized_segments.append(
            {
                "id": seg.id,
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text.strip(),
                "words": words_list,
            }
        )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_segments, f, ensure_ascii=False, indent=2)

    return len(serialized_segments), info.language, info.duration


class TranscriptionStage(PipelineStage):
    """
    Run faster-whisper on the extracted audio to produce word-level
    timestamped transcripts saved as JSON.
    """

    name = "Transcription"

    def can_skip(self, ctx: StageContext) -> bool:
        """Skip if valid transcript JSON already exists on disk."""
        if ctx.transcript_path is None or not ctx.transcript_path.exists():
            return False
        try:
            return ctx.transcript_path.stat().st_size > 0
        except OSError:
            return False

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Starting transcription", ctx.episode_id)

        # 1. Resolve audio file path (prefer extracted_audio, fallback to source_file)
        if ctx.extracted_audio is not None and ctx.extracted_audio.exists():
            audio_path = ctx.extracted_audio
        elif ctx.source_file.exists():
            audio_path = ctx.source_file
        else:
            msg = f"No valid audio or video file found for transcription: {ctx.source_file}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        output_json = ctx.work_dir / f"transcript_{ctx.episode_id}.json"

        logger.debug(
            "[Episode {}] Transcribing '{}' (language={}) -> '{}'",
            ctx.episode_id,
            audio_path,
            ctx.source_language or "auto",
            output_json,
        )

        try:
            segment_count, detected_lang, duration = await asyncio.to_thread(
                _transcribe_audio_sync,
                audio_path,
                output_json,
                ctx.source_language,
            )
        except Exception as exc:
            msg = f"Transcription error: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not output_json.exists() or output_json.stat().st_size == 0:
            msg = f"Transcription produced empty or missing transcript file: {output_json}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        # Update context
        ctx.transcript_path = output_json
        if not ctx.source_language and detected_lang:
            ctx.source_language = detected_lang

        ctx.metadata["transcription_info"] = {
            "duration": duration,
            "segment_count": segment_count,
            "language": detected_lang,
        }

        msg = (
            f"Transcription completed successfully: {segment_count} segments "
            f"({duration:.1f}s audio) saved to {output_json}"
        )
        logger.info("[Episode {}] {}", ctx.episode_id, msg)

        return StageResult(
            success=True,
            message=msg,
            context_updates={
                "transcript_path": output_json,
                "source_language": ctx.source_language,
            },
        )
