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

    async def _adjust_audio_tempo(
        self,
        input_wav: Path,
        output_wav: Path,
        speed_factor: float,
        sample_rate: int = 16000,
    ) -> bool:
        """
        Adjust audio playback tempo using FFmpeg atempo filter without changing pitch.
        Chains filters if speed_factor is outside FFmpeg's 0.5 to 2.0 single filter limit.
        """
        import shutil  # noqa: PLC0415

        if abs(speed_factor - 1.0) < 0.02:
            try:
                shutil.copy2(input_wav, output_wav)
                return True
            except OSError:
                return False

        filters: list[str] = []
        factor = float(speed_factor)
        while factor > 2.0:
            filters.append("atempo=2.0")
            factor /= 2.0
        while factor < 0.5:
            filters.append("atempo=0.5")
            factor /= 0.5
        filters.append(f"atempo={factor:.4f}")
        filter_str = ",".join(filters)

        args = [
            "-y",
            "-i",
            str(input_wav),
            "-filter:a",
            filter_str,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "wav",
            str(output_wav),
        ]
        code, _, stderr = await run_ffmpeg(args)
        if code == 0 and output_wav.exists() and output_wav.stat().st_size > 44:
            return True
        logger.warning(
            "Tempo adjustment failed for speed_factor={:.2f}: {}",
            speed_factor,
            stderr[:200],
        )
    @staticmethod
    def _strip_audio_silence(
        samples: np.ndarray,
        sample_rate: int = 16000,
        threshold: int = 250,
        pad_ms: int = 30,
    ) -> np.ndarray:
        """
        Strip leading and trailing dead silence from synthesized audio chunk,
        leaving a clean 30ms cushion to ensure consonants are not clipped.
        """
        if len(samples) == 0:
            return samples
        import numpy as np  # noqa: PLC0415

        abs_s = np.abs(samples)
        non_zero = np.where(abs_s > threshold)[0]
        if len(non_zero) == 0:
            return samples
        pad_samples = int(round((pad_ms / 1000.0) * sample_rate))
        start_idx = max(0, non_zero[0] - pad_samples)
        end_idx = min(len(samples), non_zero[-1] + 1 + pad_samples)
        return samples[start_idx:end_idx]

    async def run(self, ctx: StageContext) -> StageResult:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            logger.info(
                "[Episode {}] Skipping TTS Generation (output_mode=subtitles_only)",
                ctx.episode_id,
            )
            return StageResult(success=True, message="TTS skipped per output mode setting")

        logger.info(
            "[Episode {}] Generating Khmer TTS audio (backend={}, dynamic_speed={})",
            ctx.episode_id,
            settings.tts_backend,
            settings.tts_dynamic_speed,
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

        # Determine total duration of the master dialogue timeline
        total_samples = 0
        if ctx.extracted_audio and ctx.extracted_audio.exists():
            try:
                with wave.open(str(ctx.extracted_audio), "rb") as w_ext:
                    ext_sr = w_ext.getframerate()
                    ext_frames = w_ext.getnframes()
                    total_samples = int(round(ext_frames * (sample_rate / ext_sr)))
            except Exception as exc:
                logger.debug("Could not read frame count from extracted audio: {}", exc)

        transcription_info = ctx.metadata.get("transcription_info", {})
        meta_dur = float(transcription_info.get("duration") or 0.0)
        if meta_dur > 0:
            total_samples = max(total_samples, int(round(meta_dur * sample_rate)))

        max_seg_end = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
        total_samples = max(total_samples, int(round((max_seg_end + 3.0) * sample_rate)))
        if total_samples <= 0:
            total_samples = sample_rate * 5  # Fallback 5s buffer

        import numpy as np  # noqa: PLC0415

        master_timeline = np.zeros(total_samples, dtype=np.int16)

        # Identify all segments with valid text
        active_segments: list[tuple[int, dict]] = []
        for idx, seg in enumerate(segments):
            text = str(seg.get("translated_text", "")).strip() or str(seg.get("text", "")).strip()
            if text:
                active_segments.append((idx, seg))

        total_chunks = len(active_segments)
        successful_chunks = 0
        last_speech_end = 0.0

        for seq_i, (orig_idx, seg) in enumerate(active_segments):
            text = str(seg.get("translated_text", "")).strip() or str(seg.get("text", "")).strip()
            start_sec = max(0.0, float(seg.get("start", 0.0)))
            end_sec = max(start_sec + 0.2, float(seg.get("end", start_sec + 0.8)))
            orig_duration = max(0.2, end_sec - start_sec)

            # Look ahead to next active segment to find speech boundary
            if seq_i + 1 < len(active_segments):
                next_start_sec = max(
                    start_sec,
                    float(active_segments[seq_i + 1][1].get("start", start_sec + orig_duration)),
                )
                gap_to_next = max(orig_duration, next_start_sec - start_sec - 0.05)
            else:
                gap_to_next = orig_duration * 1.5

            # Calculate safe pre-gap lead-in shift (up to 0.12s) to relieve pressure on tight segments
            pre_gap = max(0.0, start_sec - last_speech_end)
            max_pre_shift = min(0.12, max(0.0, pre_gap - 0.04)) if pre_gap >= 0.10 else 0.0

            raw_chunk_wav = chunks_dir / f"temp_{orig_idx:04d}_raw.wav"
            final_chunk_path = chunks_dir / f"chunk_{orig_idx:04d}.wav"
            gender = str(seg.get("gender", "")).lower()
            if settings.tts_voice == "km-KH-PisethNeural":
                voice_for_seg = "km-KH-PisethNeural"
            elif settings.tts_voice == "km-KH-SreymomNeural":
                voice_for_seg = "km-KH-SreymomNeural"
            else:
                voice_for_seg = seg.get("tts_voice") or (
                    "km-KH-SreymomNeural" if gender == "female" else "km-KH-PisethNeural"
                )

            success = await self._synthesize_chunk(text, raw_chunk_wav, voice_override=voice_for_seg)
            if success and raw_chunk_wav.exists():
                try:
                    # Strip dead leading & trailing silence so Edge TTS dead air is not counted in speech duration
                    with wave.open(str(raw_chunk_wav), "rb") as in_wav:
                        raw_frames_data = in_wav.readframes(in_wav.getnframes())
                        raw_samples = np.frombuffer(raw_frames_data, dtype=np.int16)

                    trimmed_samples = self._strip_audio_silence(raw_samples, sample_rate=sample_rate)
                    if 0 < len(trimmed_samples) < len(raw_samples):
                        with wave.open(str(raw_chunk_wav), "wb") as out_wav:
                            out_wav.setnchannels(1)
                            out_wav.setsampwidth(2)
                            out_wav.setframerate(sample_rate)
                            out_wav.writeframes(trimmed_samples.tobytes())
                        raw_duration = len(trimmed_samples) / float(sample_rate)
                    else:
                        raw_duration = len(raw_samples) / float(sample_rate)

                    actual_start_sec = start_sec
                    speed_adjusted = False

                    if settings.tts_dynamic_speed and raw_duration > orig_duration:
                        # Total available window without overlapping next speaker
                        total_window = gap_to_next + max_pre_shift

                        if raw_duration <= total_window:
                            # Fits comfortably within available silence window before next speaker
                            # Apply gentle alignment (max 1.08x) so speech remains completely natural
                            if raw_duration <= gap_to_next:
                                speed_ratio = min(1.08, max(1.0, raw_duration / max(orig_duration, orig_duration * 1.25)))
                            else:
                                pre_shift = min(max_pre_shift, raw_duration - gap_to_next)
                                actual_start_sec = max(0.0, start_sec - pre_shift)
                                speed_ratio = 1.0
                        else:
                            # Speech is longer than the silence gap: use full pre-shift and clamp speedup
                            actual_start_sec = max(0.0, start_sec - max_pre_shift)
                            target_time = max(orig_duration, total_window)
                            ideal_ratio = raw_duration / target_time
                            max_speedup = float(getattr(settings, "tts_max_speedup", 1.25))
                            # Clamp strictly to max_speedup so speech never rushes or sounds unintelligible
                            speed_ratio = max(1.0, min(ideal_ratio, max_speedup))

                        if speed_ratio >= 1.04:
                            adjusted = await self._adjust_audio_tempo(
                                raw_chunk_wav,
                                final_chunk_path,
                                speed_ratio,
                                sample_rate=sample_rate,
                            )
                            if adjusted and final_chunk_path.exists():
                                speed_adjusted = True
                                logger.debug(
                                    "[Episode {}] Line #{:04d} natural speed adjusted: raw={:.2f}s -> ratio={:.2f}x (max={:.2f}x)",
                                    ctx.episode_id,
                                    orig_idx,
                                    raw_duration,
                                    speed_ratio,
                                    getattr(settings, "tts_max_speedup", 1.25),
                                )

                    if not speed_adjusted:
                        import shutil
                        shutil.copy2(raw_chunk_wav, final_chunk_path)

                    try:
                        raw_chunk_wav.unlink(missing_ok=True)
                    except OSError:
                        pass

                    # Load audio samples of final chunk
                    with wave.open(str(final_chunk_path), "rb") as in_wav:
                        frames_data = in_wav.readframes(in_wav.getnframes())
                        chunk_samples = np.frombuffer(frames_data, dtype=np.int16)

                    # Over-speak protection: Prevent dialogue from overlapping next speaker's start time
                    if seq_i + 1 < len(active_segments):
                        allowed_duration = max(0.1, next_start_sec - actual_start_sec - 0.02)
                        max_allowed_samples = int(round(allowed_duration * sample_rate))
                        if len(chunk_samples) > max_allowed_samples:
                            chunk_samples = chunk_samples[:max_allowed_samples].copy()
                            fade_len = min(len(chunk_samples), int(0.035 * sample_rate))
                            if fade_len > 0:
                                fade = np.linspace(1.0, 0.0, fade_len)
                                chunk_samples[-fade_len:] = (chunk_samples[-fade_len:] * fade).astype(np.int16)

                    # Anchor precisely at actual_start_sec
                    start_sample = int(round(actual_start_sec * sample_rate))
                    chunk_len = len(chunk_samples)

                    # Expand master timeline if needed
                    if start_sample + chunk_len > len(master_timeline):
                        pad_len = (start_sample + chunk_len) - len(master_timeline)
                        master_timeline = np.pad(master_timeline, (0, pad_len), mode="constant")

                    # Mix cleanly into timeline at the exact timestamp
                    timeline_slice = master_timeline[start_sample : start_sample + chunk_len]
                    mixed = np.clip(
                        timeline_slice.astype(np.int32) + chunk_samples.astype(np.int32),
                        -32768,
                        32767,
                    ).astype(np.int16)
                    master_timeline[start_sample : start_sample + chunk_len] = mixed
                    successful_chunks += 1
                    last_speech_end = (start_sample + chunk_len) / float(sample_rate)
                except Exception as proc_exc:
                    logger.warning("Failed processing TTS chunk #{}: {}", orig_idx, proc_exc)

            if len(active_segments) > 0:
                ctx.report_progress(int(((seq_i + 1) / len(active_segments)) * 100))

        if total_chunks > 0 and successful_chunks == 0:
            msg = f"Khmer TTS generation failed: 0 of {total_chunks} dialogue lines were synthesized"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            if final_tts_wav.exists():
                try:
                    final_tts_wav.unlink()
                except OSError:
                    pass
            return StageResult(success=False, message=msg)

        # Write master dialogue timeline to WAV file
        with wave.open(str(final_tts_wav), "wb") as out_wav:
            out_wav.setnchannels(1)
            out_wav.setsampwidth(bytes_per_sample)
            out_wav.setframerate(sample_rate)
            out_wav.writeframes(master_timeline.tobytes())

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
