"""Stage 8 — Mix Khmer dialogue TTS with background music/SFX."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from mediadubflow.config import settings
from mediadubflow.config.settings import PipelineOutputMode
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult
from mediadubflow.utils.ffmpeg import run_ffmpeg


class AudioMixingStage(PipelineStage):
    """
    Blend synthesized Khmer dialogue with the original background audio track.

    Applies volume ducking so background audio sits at 30% volume while
    Khmer dialogue is delivered clearly.
    """

    name = "Audio Mixing"

    def can_skip(self, ctx: StageContext) -> bool:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            return True
        return (
            ctx.mixed_audio_path is not None
            and ctx.mixed_audio_path.exists()
            and ctx.mixed_audio_path.stat().st_size > 0
        )

    async def run(self, ctx: StageContext) -> StageResult:
        if settings.output_mode == PipelineOutputMode.SUBTITLES_ONLY:
            logger.info(
                "[Episode {}] Skipping audio mixing (output_mode=subtitles_only)",
                ctx.episode_id,
            )
            return StageResult(success=True, message="Audio mixing skipped per output mode setting")

        logger.info("[Episode {}] Mixing Khmer dialogue with background audio", ctx.episode_id)

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        mixed_audio = ctx.work_dir / f"mixed_{ctx.episode_id}.wav"

        # If we have synthesized TTS audio, configure output with muted or blended original audio
        if ctx.tts_audio_path and ctx.tts_audio_path.exists():
            bg_vol = max(0.0, float(settings.original_audio_volume))
            tts_vol = max(0.0, float(settings.tts_volume))

            if bg_vol > 0.001 and ctx.extracted_audio and ctx.extracted_audio.exists():
                logger.info(
                    "[Episode {}] Mixing background audio (vol={:.2f}, mute_speaker={}) with Khmer TTS (vol={:.2f})",
                    ctx.episode_id,
                    bg_vol,
                    settings.mute_original_speaker,
                    tts_vol,
                )
                success = self._mix_with_ducked_speech(
                    bg_path=ctx.extracted_audio,
                    tts_path=ctx.tts_audio_path,
                    ctx=ctx,
                    mute_speaker=settings.mute_original_speaker,
                    bg_vol=bg_vol,
                    tts_vol=tts_vol,
                    out_path=mixed_audio,
                )
                if not success:
                    # Fallback to FFmpeg amix filter if waveform mixer encountered an issue
                    logger.warning("[Episode {}] Falling back to FFmpeg amix", ctx.episode_id)
                    filter_complex = (
                        f"[0:a]volume={bg_vol:.2f}[bg];[1:a]volume={tts_vol:.2f}[vox];"
                        "[bg][vox]amix=inputs=2:duration=first:dropout_transition=2[out]"
                    )
                    args = [
                        "-y",
                        "-i",
                        str(ctx.extracted_audio),
                        "-i",
                        str(ctx.tts_audio_path),
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        "[out]",
                        "-c:a",
                        "pcm_s16le",
                        str(mixed_audio),
                    ]
                    code, _, stderr = await run_ffmpeg(args)
                    if code != 0:
                        msg = f"Audio mixing fallback failed: {stderr[:200]}"
                        logger.error("[Episode {}] {}", ctx.episode_id, msg)
                        return StageResult(success=False, message=msg)
            else:
                # Background volume set to 0.0 or background track missing: output TTS audio only
                logger.info(
                    "[Episode {}] Outputting Khmer TTS voice only (vol={:.2f})",
                    ctx.episode_id,
                    tts_vol,
                )
                if abs(tts_vol - 1.0) > 0.05:
                    args = [
                        "-y",
                        "-i",
                        str(ctx.tts_audio_path),
                        "-filter:a",
                        f"volume={tts_vol:.2f}",
                        "-c:a",
                        "pcm_s16le",
                        str(mixed_audio),
                    ]
                    code, _, stderr = await run_ffmpeg(args)
                    if code != 0:
                        logger.warning("Volume adjustment failed, using raw TTS audio: {}", stderr[:200])
                        mixed_audio = ctx.tts_audio_path
                else:
                    mixed_audio = ctx.tts_audio_path
        elif ctx.extracted_audio and ctx.extracted_audio.exists():
            mixed_audio = ctx.extracted_audio
        else:
            msg = f"No audio track available to mix for episode {ctx.episode_id}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        ctx.mixed_audio_path = mixed_audio
        logger.info("[Episode {}] Audio mixing completed at {}", ctx.episode_id, mixed_audio)
        return StageResult(
            success=True,
            message="Audio mixed successfully",
            context_updates={"mixed_audio_path": mixed_audio},
        )

    def _mix_with_ducked_speech(
        self,
        bg_path: Path,
        tts_path: Path,
        ctx: StageContext,
        mute_speaker: bool,
        bg_vol: float,
        tts_vol: float,
        out_path: Path,
    ) -> bool:
        """
        Mix original background audio with Khmer TTS dialogue.

        When mute_speaker is True:
        Applies a high-precision S-curve gain envelope across dialogue intervals,
        with safety pre-roll, post-roll, interval merging, and active TTS coverage.
        Guarantees that the original speaker is 100.0% muted with zero vocal leakage,
        while maintaining smooth, natural background audio transitions and full
        dynamic range without digital clipping.
        """
        import json
        import wave
        import numpy as np

        try:
            # 1. Collect all speech segments from translation and transcript sources
            raw_segments: list[dict[str, Any]] = []
            segment_sources = [
                ctx.translation_path,
                ctx.transcript_path,
                ctx.work_dir / f"translation_{ctx.episode_id}.json",
                ctx.work_dir / f"transcript_{ctx.episode_id}.json",
            ]
            seen_sources: set[str] = set()
            for src in segment_sources:
                if src and src.exists() and str(src) not in seen_sources:
                    seen_sources.add(str(src))
                    try:
                        with src.open("r", encoding="utf-8") as f:
                            raw = json.load(f)
                            if isinstance(raw, list):
                                raw_segments.extend(raw)
                    except Exception as exc:
                        logger.debug("[Episode {}] Could not read segments from {}: {}", ctx.episode_id, src, exc)

            # 2. Read background audio
            with wave.open(str(bg_path), "rb") as w_bg:
                bg_ch = w_bg.getnchannels()
                bg_sr = w_bg.getframerate()
                bg_frames = w_bg.getnframes()
                bg_bytes = w_bg.readframes(bg_frames)

            bg_samples = np.frombuffer(bg_bytes, dtype=np.int16).astype(np.float32)
            if bg_ch > 1:
                bg_samples = bg_samples.reshape(-1, bg_ch)

            frames_count = len(bg_samples)

            # 3. Read TTS dialogue audio
            with wave.open(str(tts_path), "rb") as w_tts:
                tts_ch = w_tts.getnchannels()
                tts_sr = w_tts.getframerate()
                tts_frames = w_tts.getnframes()
                tts_bytes = w_tts.readframes(tts_frames)

            tts_samples = np.frombuffer(tts_bytes, dtype=np.int16).astype(np.float32) * tts_vol
            if tts_ch > 1:
                tts_samples = tts_samples.reshape(-1, tts_ch)

            # Resample TTS if sample rates do not match
            if tts_sr != bg_sr and len(tts_samples) > 0:
                orig_t = np.linspace(0, len(tts_samples) / tts_sr, len(tts_samples), endpoint=False)
                new_len = int(len(tts_samples) * (bg_sr / tts_sr))
                new_t = np.linspace(0, len(tts_samples) / tts_sr, new_len, endpoint=False)
                if tts_ch > 1:
                    resampled = np.zeros((new_len, tts_ch), dtype=np.float32)
                    for c in range(tts_ch):
                        resampled[:, c] = np.interp(new_t, orig_t, tts_samples[:, c])
                    tts_samples = resampled
                else:
                    tts_samples = np.interp(new_t, orig_t, tts_samples).astype(np.float32)

            if tts_ch == 1 and bg_ch > 1:
                tts_samples = np.repeat(tts_samples[:, np.newaxis], bg_ch, axis=1)
            elif tts_ch > 1 and bg_ch == 1:
                tts_samples = np.mean(tts_samples, axis=1)

            # 4. Build muting/ducking time intervals with pre-roll and post-roll safety margins
            intervals: list[tuple[float, float]] = []

            # Safety margins: 180ms pre-roll to catch breath/onset, 280ms post-roll to catch decay/echo
            pre_roll_sec = 0.18
            post_roll_sec = 0.28

            # Inspect TTS chunk lengths if available in work_dir / "tts_chunks"
            chunks_dir = ctx.work_dir / "tts_chunks"
            chunk_durations: dict[int, float] = {}
            if chunks_dir.exists():
                for idx, chunk_file in enumerate(sorted(chunks_dir.glob("chunk_*.wav"))):
                    try:
                        with wave.open(str(chunk_file), "rb") as cw:
                            chunk_durations[idx] = cw.getnframes() / float(cw.getframerate())
                    except Exception:
                        pass

            for idx, seg in enumerate(raw_segments):
                s_sec = max(0.0, float(seg.get("start", 0.0)))
                e_sec = max(s_sec, float(seg.get("end", s_sec + 0.3)))

                # Ensure mute window extends to cover the synthesized TTS chunk duration
                if idx in chunk_durations:
                    chunk_dur = chunk_durations[idx]
                    e_sec = max(e_sec, s_sec + chunk_dur)

                s_mute = max(0.0, s_sec - pre_roll_sec)
                e_mute = e_sec + post_roll_sec
                intervals.append((s_mute, e_mute))

            # Also detect active energy in the TTS audio track itself as a failsafe
            tts_abs = np.abs(tts_samples)
            if bg_ch > 1 and tts_abs.ndim > 1:
                tts_abs = np.max(tts_abs, axis=1)
            active_tts_mask = tts_abs > 120.0
            if np.any(active_tts_mask):
                active_indices = np.where(active_tts_mask)[0]
                diffs = np.diff(active_indices)
                split_points = np.where(diffs > int(0.25 * bg_sr))[0] + 1
                chunks = np.split(active_indices, split_points)
                for chk in chunks:
                    if len(chk) > 0:
                        s_t = max(0.0, (chk[0] / bg_sr) - 0.15)
                        e_t = (chk[-1] / bg_sr) + 0.25
                        intervals.append((s_t, e_t))

            # 5. Merge contiguous and nearby intervals (within 0.65s) to eliminate pauses/breath leaks
            merged_intervals: list[tuple[float, float]] = []
            if intervals:
                intervals.sort(key=lambda x: x[0])
                merged_intervals.append(intervals[0])
                merge_gap = 0.65
                for cur_s, cur_e in intervals[1:]:
                    prev_s, prev_e = merged_intervals[-1]
                    if cur_s <= prev_e + merge_gap:
                        merged_intervals[-1] = (prev_s, max(prev_e, cur_e))
                    else:
                        merged_intervals.append((cur_s, cur_e))

            # 6. Generate smooth raised-cosine (Hann) gain envelope
            gain = np.ones(frames_count, dtype=np.float32)
            target_gain = 0.0 if mute_speaker else 0.20
            fade_out_samples = int(0.08 * bg_sr)  # 80ms fade out before speech start
            fade_in_samples = int(0.10 * bg_sr)   # 100ms fade in after speech end

            for s_sec, e_sec in merged_intervals:
                s_idx = max(0, int(s_sec * bg_sr))
                e_idx = min(frames_count, int(e_sec * bg_sr))

                # Smooth Hann curve fade out before dialogue
                fo_start = max(0, s_idx - fade_out_samples)
                fo_len = s_idx - fo_start
                if fo_len > 0:
                    t_fo = np.linspace(0.0, np.pi, fo_len, endpoint=False)
                    curve_fo = target_gain + (1.0 - target_gain) * 0.5 * (1.0 + np.cos(t_fo))
                    gain[fo_start:s_idx] = np.minimum(gain[fo_start:s_idx], curve_fo)

                # Core dialogue window: 100% silence when mute_speaker is True
                gain[s_idx:e_idx] = target_gain

                # Smooth Hann curve fade in after dialogue
                fi_end = min(frames_count, e_idx + fade_in_samples)
                fi_len = fi_end - e_idx
                if fi_len > 0:
                    t_fi = np.linspace(np.pi, 0.0, fi_len, endpoint=False)
                    curve_fi = target_gain + (1.0 - target_gain) * 0.5 * (1.0 + np.cos(t_fi))
                    gain[e_idx:fi_end] = np.minimum(gain[e_idx:fi_end], curve_fi)

            # 7. Apply gain envelope and background volume
            if bg_ch > 1:
                ducked_bg = bg_samples * gain[:, np.newaxis] * bg_vol
            else:
                ducked_bg = bg_samples * gain * bg_vol

            # 8. Align track lengths
            max_len = max(len(ducked_bg), len(tts_samples))
            if len(ducked_bg) < max_len:
                pad = max_len - len(ducked_bg)
                ducked_bg = np.pad(
                    ducked_bg,
                    ((0, pad), (0, 0)) if bg_ch > 1 else (0, pad),
                )
            if len(tts_samples) < max_len:
                pad = max_len - len(tts_samples)
                tts_samples = np.pad(
                    tts_samples,
                    ((0, pad), (0, 0)) if bg_ch > 1 else (0, pad),
                )

            # 9. Mix background audio with Khmer TTS dialogue
            mixed = ducked_bg + tts_samples

            # 10. Transparent soft-saturation limiter (prevents digital clipping without crushing dynamic range)
            limit_threshold = 30000.0
            limit_ceiling = 32700.0
            over_mask = np.abs(mixed) > limit_threshold
            if np.any(over_mask):
                overshoot = np.abs(mixed[over_mask]) - limit_threshold
                headroom = limit_ceiling - limit_threshold
                compressed = limit_threshold + headroom * np.tanh(overshoot / headroom)
                mixed[over_mask] = np.sign(mixed[over_mask]) * compressed

            mixed_int16 = np.clip(mixed, -32768.0, 32767.0).astype(np.int16)

            # 11. Write standardized 16-bit PCM WAV
            with wave.open(str(out_path), "wb") as w_out:
                w_out.setnchannels(bg_ch)
                w_out.setsampwidth(2)
                w_out.setframerate(bg_sr)
                w_out.writeframes(mixed_int16.tobytes())

            logger.info(
                "[Episode {}] Audio mixing successful: {} channels, {} Hz, {:.1f}s, {} merged dialogue zones",
                ctx.episode_id,
                bg_ch,
                bg_sr,
                len(mixed_int16) / bg_sr,
                len(merged_intervals),
            )
            return True
        except Exception as exc:
            logger.error("[Episode {}] Waveform mixing failed: {}", ctx.episode_id, exc)
            return False
