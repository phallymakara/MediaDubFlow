"""Stage 7 — Generate Khmer speech audio using TTS."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.config.settings import PipelineOutputMode, TTSBackend
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.ffmpeg import run_ffmpeg


class TTSGenerationStage(PipelineStage):
    """
    Synthesize Khmer speech for each translated segment and align them
    onto a continuous audio timeline matching the original video.
    """

    name = "TTS Generation"

    def can_skip(self, ctx: StageContext) -> bool:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            return True
        return (
            ctx.tts_audio_path is not None
            and ctx.tts_audio_path.exists()
            and ctx.tts_audio_path.stat().st_size > 0
        )

    async def _synthesize_chunk(self, text: str, chunk_wav: Path) -> bool:
        """Synthesize a single text line to a 16kHz mono WAV file."""
        temp_audio = chunk_wav.with_suffix(".mp3")
        try:
            if settings.tts_backend == TTSBackend.OPENAI:
                if not settings.openai_api_key.strip():
                    raise ValueError("OPENAI_API_KEY is not configured for OpenAI TTS.")
                from openai import AsyncOpenAI  # noqa: PLC0415

                client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=60.0)
                response = await client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                )
                await response.astream_to_file(temp_audio)
            else:
                # Default: Edge-TTS with native Khmer neural voices
                import edge_tts  # noqa: PLC0415

                voice = settings.tts_voice or "km-KH-PisethNeural"
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(temp_audio))

            # Convert to standardized 16kHz 16-bit mono WAV using FFmpeg
            code, _, stderr = await run_ffmpeg(
                ["-y", "-i", str(temp_audio), "-ar", "16000", "-ac", "1", "-f", "wav", str(chunk_wav)]
            )
            if code != 0:
                logger.warning("FFmpeg WAV conversion failed for TTS chunk: {}", stderr[:200])
                return False
            return True
        except Exception as exc:
            logger.warning("TTS synthesis failed for line '{}': {}", text[:50], exc)
            return False
        finally:
            if temp_audio.exists():
                try:
                    temp_audio.unlink()
                except OSError:
                    pass

    async def run(self, ctx: StageContext) -> StageResult:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            logger.info(
                "[Episode {}] Skipping TTS Generation (output_mode=subtitles_only)",
                ctx.episode_id,
            )
            return StageResult(success=True, message="TTS skipped per output mode setting")

        logger.info(
            "[Episode {}] Generating Khmer TTS audio (backend={})",
            ctx.episode_id,
            settings.tts_backend,
        )

        if ctx.translation_path is None or not ctx.translation_path.exists():
            msg = f"Translation file not found for episode {ctx.episode_id}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        try:
            with ctx.translation_path.open("r", encoding="utf-8") as f:
                segments = json.load(f)
        except Exception as exc:
            msg = f"Failed to read translation JSON: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not isinstance(segments, list):
            msg = f"Translation JSON format invalid (expected list, got {type(segments)})"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = ctx.work_dir / "tts_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        final_tts_wav = ctx.work_dir / f"dialogue_{ctx.episode_id}.wav"
        sample_rate = 16000
        bytes_per_sample = 2  # 16-bit mono

        current_sample_pos = 0

        with wave.open(str(final_tts_wav), "wb") as out_wav:
            out_wav.setnchannels(1)
            out_wav.setsampwidth(bytes_per_sample)
            out_wav.setframerate(sample_rate)

            for idx, seg in enumerate(segments):
                text = str(seg.get("translated_text", "")).strip()
                if not text:
                    continue

                start_sec = float(seg.get("start", 0.0))
                target_start_sample = int(start_sec * sample_rate)

                # Pad silence up to segment start time
                if target_start_sample > current_sample_pos:
                    silence_samples = target_start_sample - current_sample_pos
                    out_wav.writeframes(b"\x00" * (silence_samples * bytes_per_sample))
                    current_sample_pos = target_start_sample

                chunk_wav = chunks_dir / f"chunk_{idx:04d}.wav"
                success = await self._synthesize_chunk(text, chunk_wav)
                if success and chunk_wav.exists():
                    try:
                        with wave.open(str(chunk_wav), "rb") as in_wav:
                            frames = in_wav.readframes(in_wav.getnframes())
                            out_wav.writeframes(frames)
                            current_sample_pos += in_wav.getnframes()
                    except Exception as wave_exc:
                        logger.warning("Could not append audio chunk {}: {}", chunk_wav, wave_exc)

        ctx.tts_audio_path = final_tts_wav
        logger.info(
            "[Episode {}] Generated full Khmer dialogue audio track at {}",
            ctx.episode_id,
            final_tts_wav,
        )
        return StageResult(
            success=True,
            message="Khmer TTS dialogue audio generated successfully",
            context_updates={"tts_audio_path": final_tts_wav},
        )
