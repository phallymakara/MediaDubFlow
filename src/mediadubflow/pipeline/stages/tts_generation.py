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
        if (
            ctx.tts_audio_path is not None
            and ctx.tts_audio_path.exists()
            and ctx.tts_audio_path.stat().st_size > 1000
        ):
            try:
                import numpy as np

                with wave.open(str(ctx.tts_audio_path), "rb") as w:
                    frames = w.readframes(min(w.getnframes(), 48000))
                    samples = np.frombuffer(frames, dtype=np.int16)
                    if len(samples) > 0 and int(np.max(np.abs(samples))) > 10:
                        return True
            except Exception:
                pass
        return False

    async def _synthesize_chunk(self, text: str, chunk_wav: Path, voice_override: str | None = None) -> bool:
        """Synthesize a single text line to a 16kHz mono WAV file with retries."""
        clean_text = text.strip()
        if not clean_text:
            return False

        temp_audio = chunk_wav.with_suffix(".mp3")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if settings.tts_backend == TTSBackend.OPENAI:
                    if not settings.openai_api_key.strip():
                        raise ValueError("OPENAI_API_KEY is not configured for OpenAI TTS.")
                    from openai import AsyncOpenAI  # noqa: PLC0415

                    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=60.0)
                    openai_voice = "nova" if (voice_override and "Sreymom" in voice_override) else "alloy"
                    response = await client.audio.speech.create(
                        model="tts-1",
                        voice=openai_voice,
                        input=clean_text,
                        speed=max(0.25, min(4.0, float(settings.tts_speed))),
                    )
                    await response.astream_to_file(temp_audio)
                else:
                    # Default: Edge-TTS with native Khmer neural voices (100% Khmer guarantee)
                    import edge_tts  # noqa: PLC0415

                    voice = voice_override or settings.tts_voice or "km-KH-PisethNeural"
                    # Enforce authentic Khmer neural voices only - never speak in Chinese
                    if "Sreymom" in str(voice_override):
                        voice = "km-KH-SreymomNeural"
                    elif "Piseth" in str(voice_override):
                        voice = "km-KH-PisethNeural"
                    elif not voice.startswith("km-KH-"):
                        voice = "km-KH-PisethNeural"

                    # Replace any remaining CJK names/terms with natural Khmer script
                    cjk_replacements = {
                        "景辰": "ជីងឆិន",
                        "霍锦成": "ហ័រជីងឆិន",
                        "穗穗": "ស៊ុយស៊ុយ",
                        "碎碎": "ស៊ុយស៊ុយ",
                        "刘非": "លីវហ្វី",
                        "傅士周": "ភូស៊ីចូវ",
                        "傅时周": "ភូស៊ីចូវ",
                        "段主": "ម្ចាស់ក្សត្រី",
                        "既然结婚了": "ក្នុងពេលដែលរៀបការរួចហើយ",
                    }
                    for cjk_k, km_v in cjk_replacements.items():
                        clean_text = clean_text.replace(cjk_k, km_v)

                    # Strip any remaining stray CJK characters so zero Chinese is ever sent to TTS
                    clean_text = "".join(c for c in clean_text if not ("\u4e00" <= c <= "\u9fff")).strip()

                    # If text still lacks Khmer script, anchor with polite Khmer particle so Khmer TTS never fails
                    if not any("\u1780" <= ch <= "\u17ff" for ch in clean_text):
                        clean_text = f"បាទ {clean_text}".strip()

                    rate_pct = int(round((settings.tts_speed - 1.0) * 100))
                    rate_str = f"{rate_pct:+d}%"
                    vol_pct = int(round((settings.tts_volume - 1.0) * 100))
                    vol_str = f"{vol_pct:+d}%"

                    # If attempt > 2, try alternate Khmer voice
                    if attempt > 2:
                        voice = "km-KH-SreymomNeural" if voice == "km-KH-PisethNeural" else "km-KH-PisethNeural"

                    communicate = edge_tts.Communicate(clean_text, voice, rate=rate_str, volume=vol_str)
                    await communicate.save(str(temp_audio))

                if temp_audio.exists() and temp_audio.stat().st_size > 0:
                    # Convert to standardized 16kHz 16-bit mono WAV using FFmpeg
                    code, _, stderr = await run_ffmpeg(
                        ["-y", "-i", str(temp_audio), "-ar", "16000", "-ac", "1", "-f", "wav", str(chunk_wav)]
                    )
                    if code == 0 and chunk_wav.exists() and chunk_wav.stat().st_size > 44:
                        return True
                    logger.warning(
                        "FFmpeg WAV conversion failed for TTS chunk (attempt {}/{}): {}",
                        attempt,
                        max_attempts,
                        stderr[:200],
                    )
            except Exception as exc:
                logger.warning(
                    "TTS synthesis failed for line '{}' (attempt {}/{}): {}",
                    clean_text[:50],
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    import asyncio

                    await asyncio.sleep(1.0 * attempt)
            finally:
                if temp_audio.exists():
                    try:
                        temp_audio.unlink()
                    except OSError:
                        pass
        return False

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
        successful_chunks = 0
        total_chunks = len([s for s in segments if str(s.get("translated_text", "")).strip()])

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
                gender = str(seg.get("gender", "")).lower()
                if settings.tts_voice == "km-KH-PisethNeural":
                    voice_for_seg = "km-KH-PisethNeural"
                elif settings.tts_voice == "km-KH-SreymomNeural":
                    voice_for_seg = "km-KH-SreymomNeural"
                else:
                    voice_for_seg = seg.get("tts_voice") or (
                        "km-KH-SreymomNeural" if gender == "female" else "km-KH-PisethNeural"
                    )
                success = await self._synthesize_chunk(text, chunk_wav, voice_override=voice_for_seg)
                if success and chunk_wav.exists():
                    try:
                        with wave.open(str(chunk_wav), "rb") as in_wav:
                            frames = in_wav.readframes(in_wav.getnframes())
                            out_wav.writeframes(frames)
                            current_sample_pos += in_wav.getnframes()
                            successful_chunks += 1
                    except Exception as wave_exc:
                        logger.warning("Failed appending TTS chunk #{}: {}", idx, wave_exc)

                if len(segments) > 0:
                    ctx.report_progress(int(((idx + 1) / len(segments)) * 100))

        if total_chunks > 0 and successful_chunks == 0:
            msg = f"Khmer TTS generation failed: 0 of {total_chunks} dialogue lines were synthesized"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            if final_tts_wav.exists():
                try:
                    final_tts_wav.unlink()
                except OSError:
                    pass
            return StageResult(success=False, message=msg)

        ctx.tts_audio_path = final_tts_wav
        logger.info(
            "[Episode {}] Generated full Khmer dialogue audio track ({}/{} lines synthesized) at {}",
            ctx.episode_id,
            successful_chunks,
            total_chunks,
            final_tts_wav,
        )
        return StageResult(
            success=True,
            message=f"Khmer TTS dialogue audio generated successfully ({successful_chunks}/{total_chunks} lines)",
            context_updates={"tts_audio_path": final_tts_wav},
        )
