"""Unit tests for TTSGenerationStage dynamic speed and timestamp anchoring."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mediadubflow.config import settings
from mediadubflow.pipeline.base import StageContext
from mediadubflow.pipeline.stages.tts_generation import TTSGenerationStage


def _create_sine_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    """Generate a valid test WAV file with a sine wave tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


@pytest.mark.asyncio
async def test_tts_generation_timestamp_anchoring_and_dynamic_speed(tmp_path: Path) -> None:
    """TTS generation anchors chunks at exact start timestamps and fits them without delay."""
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 2 speech segments:
    # Seg 0: 1.0s to 2.5s (duration 1.5s)
    # Seg 1: 3.5s to 5.0s (duration 1.5s)
    translation_data = [
        {
            "id": 1,
            "start": 1.0,
            "end": 2.5,
            "translated_text": "ជំរាបសួរអ្នកទាំងអស់គ្នា",
            "gender": "male",
        },
        {
            "id": 2,
            "start": 3.5,
            "end": 5.0,
            "translated_text": "ខ្ញុំរីករាយណាស់ដែលបានជួបអ្នក",
            "gender": "female",
        },
    ]

    trans_file = work_dir / "translation_test.json"
    trans_file.write_text(json.dumps(translation_data), encoding="utf-8")

    ctx = StageContext(
        project_id=1,
        episode_id=1,
        source_file=tmp_path / "input.mp4",
        output_dir=tmp_path / "output",
        work_dir=work_dir,
        metadata={"transcription_info": {"duration": 7.0}},
        translation_path=trans_file,
    )

    stage = TTSGenerationStage()

    # Mock synthesize_chunk to create a 2.0s WAV for each line (which is longer than 1.5s original duration)
    async def mock_synth(text: str, chunk_wav: Path, voice_override: str | None = None) -> bool:
        _create_sine_wav(chunk_wav, duration_sec=2.0, sample_rate=16000)
        return True

    # Mock adjust_audio_tempo to create a sped-up 1.4s WAV
    async def mock_tempo(input_wav: Path, output_wav: Path, speed_factor: float, sample_rate: int = 16000) -> bool:
        assert speed_factor > 1.0
        _create_sine_wav(output_wav, duration_sec=1.4, sample_rate=sample_rate)
        return True

    with (
        patch.object(stage, "_synthesize_chunk", side_effect=mock_synth),
        patch.object(stage, "_adjust_audio_tempo", side_effect=mock_tempo),
    ):
        result = await stage.run(ctx)

        assert result.success is True
        assert ctx.tts_audio_path is not None
        assert ctx.tts_audio_path.exists()

        with wave.open(str(ctx.tts_audio_path), "rb") as w:
            total_frames = w.getnframes()
            sr = w.getframerate()
            assert sr == 16000
            samples = np.frombuffer(w.readframes(total_frames), dtype=np.int16)

            # Seg 0 starts at 1.0s (sample 16000). Before sample 16000 should be silence (zeros)
            start_seg0 = int(1.0 * 16000)
            assert np.all(samples[:start_seg0] == 0)

            # At and after sample 16000, there should be dialogue audio
            assert np.max(np.abs(samples[start_seg0 : start_seg0 + 1000])) > 100

            # Seg 1 starts at 3.5s (sample 56000)
            start_seg1 = int(3.5 * 16000)
            # Before seg 1 starts, there should be silence between ~2.5s and 3.4s
            pause_samples = samples[int(2.6 * 16000) : start_seg1]
            assert np.all(pause_samples == 0)

            # Seg 1 has audio starting right at 3.5s
            assert np.max(np.abs(samples[start_seg1 : start_seg1 + 1000])) > 100


@pytest.mark.asyncio
async def test_tts_generation_adjust_audio_tempo_helper(tmp_path: Path) -> None:
    """_adjust_audio_tempo invokes run_ffmpeg with correctly chained atempo filters."""
    stage = TTSGenerationStage()
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    _create_sine_wav(input_wav, duration_sec=1.0)

    # Mock run_ffmpeg returning success and creating output
    async def mock_run_ffmpeg(args):
        # Create output file to simulate successful conversion
        _create_sine_wav(output_wav, duration_sec=0.5)
        return 0, "", ""

    with patch("mediadubflow.pipeline.stages.tts_generation.run_ffmpeg", side_effect=mock_run_ffmpeg) as mock_ffmpeg:
        # Test factor 2.4 (should chain atempo=2.0,atempo=1.2000)
        success = await stage._adjust_audio_tempo(input_wav, output_wav, speed_factor=2.4)
        assert success is True
        mock_ffmpeg.assert_called_once()
        called_args = mock_ffmpeg.call_args[0][0]
        filter_idx = called_args.index("-filter:a")
        assert "atempo=2.0" in called_args[filter_idx + 1]
        assert "atempo=1.2" in called_args[filter_idx + 1]
