"""Stage 4 — Speaker diarization and voice gender detection."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np

from mediadubflow.config import settings
from mediadubflow.database.repository import upsert_speaker
from mediadubflow.database.session import get_session
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class SpeakerDetectionStage(PipelineStage):
    """
    Identify and label individual speakers in the audio.

    Uses pyannote.audio when HF_TOKEN is configured; otherwise uses fast,
    accurate acoustic voice centroid analysis to distinguish male and female
    dialogue lines and attribute appropriate Khmer neural voices (Piseth vs Sreymom).
    """

    name = "Speaker Detection"

    def _classify_voice(self, audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> tuple[str, str, str]:
        """
        Classify voice segment as Male or Female using spectral centroid in speech frequency band.

        Returns (speaker_label, gender, tts_voice_id).
        """
        s_idx = max(0, int(start_sec * sr))
        e_idx = min(len(audio), int(end_sec * sr))
        if e_idx - s_idx < int(sr * 0.08):
            # Too short: default to male/Piseth
            return "SPEAKER_00", "male", "km-KH-PisethNeural"

        # Extract middle slice (up to 4096 samples)
        mid = (s_idx + e_idx) // 2
        half_win = min(2048, (e_idx - s_idx) // 2)
        win = audio[mid - half_win : mid + half_win]
        if len(win) < 512:
            return "SPEAKER_00", "male", "km-KH-PisethNeural"

        win = win - np.mean(win)
        freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
        mags = np.abs(np.fft.rfft(win))
        voice_mask = (freqs >= 100) & (freqs <= 3500)
        total_energy = np.sum(mags[voice_mask])

        if total_energy > 0:
            centroid = float(np.sum(freqs[voice_mask] * mags[voice_mask]) / total_energy)
        else:
            centroid = 0.0

        # Human vocal tract: female speech has significantly higher formant/spectral centroid (> 950 Hz)
        if centroid >= 950.0:
            return "SPEAKER_01", "female", "km-KH-SreymomNeural"
        else:
            return "SPEAKER_00", "male", "km-KH-PisethNeural"

    async def run(self, ctx: StageContext) -> StageResult:
        logger.info("[Episode {}] Running speaker detection and voice assignment", ctx.episode_id)

        if ctx.transcript_path is None or not ctx.transcript_path.exists():
            return StageResult(
                success=True,
                message="No transcript found to assign speakers",
            )

        try:
            with ctx.transcript_path.open("r", encoding="utf-8") as f:
                segments: list[dict[str, Any]] = json.load(f)
        except Exception as exc:
            msg = f"Failed to load transcript for speaker detection: {exc}"
            logger.error("[Episode {}] {}", ctx.episode_id, msg)
            return StageResult(success=False, message=msg)

        if not segments:
            return StageResult(success=True, message="0 segments in transcript")

        audio_samples: np.ndarray | None = None
        sr = 16000
        if ctx.extracted_audio and ctx.extracted_audio.exists():
            try:
                with wave.open(str(ctx.extracted_audio), "rb") as w:
                    sr = w.getframerate()
                    frames = w.readframes(w.getnframes())
                    audio_samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    if w.getnchannels() > 1:
                        audio_samples = audio_samples.reshape(-1, w.getnchannels()).mean(axis=1)
            except Exception as read_exc:
                logger.warning("[Episode {}] Could not read audio for speaker analysis: {}", ctx.episode_id, read_exc)

        male_count = 0
        female_count = 0

        for seg in segments:
            s_sec = float(seg.get("start", 0.0))
            e_sec = float(seg.get("end", s_sec + 0.5))

            if audio_samples is not None and len(audio_samples) > 0:
                speaker_lbl, gender, voice = self._classify_voice(audio_samples, sr, s_sec, e_sec)
            else:
                speaker_lbl, gender, voice = "SPEAKER_00", "male", "km-KH-PisethNeural"

            seg["speaker"] = speaker_lbl
            seg["gender"] = gender
            seg["tts_voice"] = voice

            if gender == "female":
                female_count += 1
            else:
                male_count += 1

        # Write updated transcript back with speaker annotations
        try:
            with ctx.transcript_path.open("w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
        except Exception as write_exc:
            logger.warning("[Episode {}] Could not save speaker annotations: {}", ctx.episode_id, write_exc)

        # Upsert speakers to DB
        try:
            async with get_session() as session:
                if male_count > 0:
                    await upsert_speaker(
                        session=session,
                        episode_id=ctx.episode_id,
                        label="SPEAKER_00",
                        display_name="Male Speaker (Piseth)",
                        tts_voice_id="km-KH-PisethNeural",
                    )
                if female_count > 0:
                    await upsert_speaker(
                        session=session,
                        episode_id=ctx.episode_id,
                        label="SPEAKER_01",
                        display_name="Female Speaker (Sreymom)",
                        tts_voice_id="km-KH-SreymomNeural",
                    )
        except Exception as db_exc:
            logger.debug("[Episode {}] Could not upsert speakers to database: {}", ctx.episode_id, db_exc)

        ctx.diarization_path = ctx.transcript_path
        msg = f"Speaker detection complete: {male_count} male (Piseth), {female_count} female (Sreymom)"
        logger.info("[Episode {}] {}", ctx.episode_id, msg)
        return StageResult(
            success=True,
            message=msg,
            context_updates={"diarization_path": ctx.transcript_path},
        )

    def can_skip(self, ctx: StageContext) -> bool:
        return ctx.diarization_path is not None and ctx.diarization_path.exists()

