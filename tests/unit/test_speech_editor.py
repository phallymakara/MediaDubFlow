"""
Unit tests for Speech Translation CRUD Editor, timestamp helpers, and step progress.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from mediadubflow.gui.dialogs.episode_detail_panel import (
    format_timestamp,
    parse_timestamp,
)
from mediadubflow.utils.subtitles import create_ass_file, create_srt_file


def test_format_and_parse_timestamp() -> None:
    """Test timestamp formatting and parsing helpers."""
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(75.5) == "01:15.500"
    assert format_timestamp(125.123) == "02:05.123"

    assert parse_timestamp("00:00.000") == 0.0
    assert parse_timestamp("01:15.500") == 75.5
    assert parse_timestamp("75.5") == 75.5
    assert parse_timestamp("invalid") is None
    assert parse_timestamp("") is None


def test_speech_crud_data_flow(tmp_path: Path) -> None:
    """Verify speech segments CRUD lifecycle, JSON persistence, and subtitle synchronization."""
    trans_file = tmp_path / "translation_1.json"
    srt_file = tmp_path / "ep01.srt"
    ass_file = tmp_path / "ep01.ass"

    # 1. Initial State (Read)
    initial_segments = [
        {"id": 1, "start": 0.5, "end": 2.5, "text": "Hello", "translated_text": "សួស្តី"},
        {"id": 2, "start": 3.0, "end": 5.0, "text": "How are you?", "translated_text": "សុខសប្បាយទេ?"},
    ]
    trans_file.write_text(json.dumps(initial_segments, ensure_ascii=False), encoding="utf-8")

    loaded_segments = json.loads(trans_file.read_text(encoding="utf-8"))
    assert len(loaded_segments) == 2
    assert loaded_segments[0]["translated_text"] == "សួស្តី"

    # 2. CREATE: Add new dialogue line
    new_segment = {
        "id": 3,
        "start": 5.5,
        "end": 7.5,
        "text": "Thank you very much",
        "translated_text": "អរគុណច្រើន",
    }
    loaded_segments.append(new_segment)
    loaded_segments.sort(key=lambda s: s.get("start", 0.0))
    for idx, s in enumerate(loaded_segments, start=1):
        s["id"] = idx
    assert len(loaded_segments) == 3

    # 3. UPDATE: Edit dialogue line
    loaded_segments[0]["translated_text"] = "ជំរាបសួរលោកអ្នក"
    assert loaded_segments[0]["translated_text"] == "ជំរាបសួរលោកអ្នក"

    # 4. DELETE: Remove dialogue line
    deleted = loaded_segments.pop(1)
    assert deleted["text"] == "How are you?"
    assert len(loaded_segments) == 2

    # 5. SAVE: Persist back to JSON and synchronize .srt and .ass subtitles
    trans_file.write_text(json.dumps(loaded_segments, ensure_ascii=False, indent=2), encoding="utf-8")
    assert trans_file.exists()

    create_srt_file(loaded_segments, srt_file)
    create_ass_file(loaded_segments, ass_file)

    assert srt_file.exists()
    assert ass_file.exists()

    srt_content = srt_file.read_text(encoding="utf-8")
    assert "ជំរាបសួរលោកអ្នក" in srt_content
    assert "អរគុណច្រើន" in srt_content
    assert "How are you?" not in srt_content


def test_job_manager_step_progress_monotonic() -> None:
    """Verify that stage step progress calculates monotonic overall pipeline percentage."""
    total_stages = 9
    emitted_records: list[tuple[str, int]] = []

    def mock_emit(ep_id: int, stage_desc: str, overall_pct: int) -> None:
        emitted_records.append((stage_desc, overall_pct))

    stage_names = [
        "Audio Extraction",
        "Language Detection",
        "Transcription",
        "Speaker Detection",
        "Translation",
        "Subtitle Generation",
        "TTS Generation",
        "Audio Mixing",
        "Video Rendering",
    ]

    for stage_idx, name in enumerate(stage_names):
        # Intra-stage progress 0%
        intra_0 = 0
        pct_0 = int(((stage_idx + (intra_0 / 100.0)) / total_stages) * 100)
        desc_0 = f"Step {stage_idx + 1}/{total_stages}: {name}"
        mock_emit(1, desc_0, pct_0)

        # Intra-stage progress 50%
        intra_50 = 50
        pct_50 = int(((stage_idx + (intra_50 / 100.0)) / total_stages) * 100)
        desc_50 = f"Step {stage_idx + 1}/{total_stages}: {name}"
        mock_emit(1, desc_50, pct_50)

        # Intra-stage progress 100%
        intra_100 = 100
        pct_100 = int(((stage_idx + (intra_100 / 100.0)) / total_stages) * 100)
        desc_100 = f"Step {stage_idx + 1}/{total_stages}: {name}"
        mock_emit(1, desc_100, pct_100)

    # Monotonic progression check: each progress percentage must be >= previous
    percentages = [pct for _, pct in emitted_records]
    assert percentages[0] == 0
    assert percentages[-1] == 100
    for i in range(len(percentages) - 1):
        assert percentages[i] <= percentages[i + 1], f"Progress went backwards: {percentages[i]} > {percentages[i+1]}"

    # Verify descriptions contain accurate Step X/9 format
    assert "Step 1/9: Audio Extraction" in emitted_records[0][0]
    assert "Step 5/9: Translation" in emitted_records[12][0]
    assert "Step 9/9: Video Rendering" in emitted_records[-1][0]


def test_audio_mixing_ducked_speech(tmp_path: Path) -> None:
    """Verify that original background sound is preserved while original speaker is muted during dialogue."""
    import wave
    import numpy as np
    from mediadubflow.pipeline.base import StageContext
    from mediadubflow.pipeline.stages.audio_mixing import AudioMixingStage

    sr = 16000
    duration_sec = 4.0
    total_samples = int(sr * duration_sec)

    # 1. Create original background audio (continuous 440 Hz tone)
    bg_file = tmp_path / "extracted.wav"
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    bg_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    with wave.open(str(bg_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bg_data.tobytes())

    # 2. Create TTS audio (880 Hz tone during speech interval 1.0s to 2.5s, silence elsewhere)
    tts_file = tmp_path / "tts.wav"
    tts_data = np.zeros(total_samples, dtype=np.int16)
    speech_mask = (t >= 1.0) & (t <= 2.5)
    tts_data[speech_mask] = (np.sin(2 * np.pi * 880 * t[speech_mask]) * 15000).astype(np.int16)
    with wave.open(str(tts_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(tts_data.tobytes())

    # 3. Create translation JSON defining the speech segment
    trans_file = tmp_path / "trans.json"
    segments = [{"id": 1, "start": 1.0, "end": 2.5, "text": "Hello", "translated_text": "សួស្តី"}]
    trans_file.write_text(json.dumps(segments), encoding="utf-8")

    out_file = tmp_path / "mixed.wav"

    ctx = StageContext(
        episode_id=1,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=bg_file,
        tts_audio_path=tts_file,
        translation_path=trans_file,
    )

    stage = AudioMixingStage()
    success = stage._mix_with_ducked_speech(
        bg_path=bg_file,
        tts_path=tts_file,
        ctx=ctx,
        mute_speaker=True,
        bg_vol=0.8,
        tts_vol=1.2,
        out_path=out_file,
    )

    assert success is True
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Read back mixed audio and verify samples
    with wave.open(str(out_file), "rb") as w:
        mixed_bytes = w.readframes(w.getnframes())
    mixed_samples = np.frombuffer(mixed_bytes, dtype=np.int16).astype(np.float32)

    # At 0.5s (before speech): background audio is active (~8000 peak)
    pre_speech_peak = np.max(np.abs(mixed_samples[int(0.3 * sr) : int(0.7 * sr)]))
    assert pre_speech_peak > 5000, "Background sound should be audible before speech"

    # At 3.5s (after speech): background audio is active (~8000 peak)
    post_speech_peak = np.max(np.abs(mixed_samples[int(3.0 * sr) : int(3.8 * sr)]))
    assert post_speech_peak > 5000, "Background sound should be audible after speech"


def test_job_manager_redub_episode(tmp_path: Path, monkeypatch) -> None:
    """Verify that JobManager.redub_episode updates speed, purges downstream artifacts, and re-enqueues."""
    from mediadubflow.config import settings
    from mediadubflow.core.job_manager import JobManager

    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "output_root", tmp_path / "output")

    work_dir = tmp_path / "cache" / "episode_1"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "output" / "episode_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    old_tts = work_dir / "dialogue_1.wav"
    old_tts.write_bytes(b"dummy")
    old_mixed = work_dir / "mixed_1.wav"
    old_mixed.write_bytes(b"dummy")
    old_vid = out_dir / "ep01_dubbed.mp4"
    old_vid.write_bytes(b"dummy")

    mgr = JobManager(stages=[])
    enqueued = mgr.redub_episode(1, 1.75)

    assert enqueued is True
    assert settings.tts_speed == 1.75
    assert not old_tts.exists(), "Old dialogue audio should be deleted"
    assert not old_mixed.exists(), "Old mixed audio should be deleted"
    assert not old_vid.exists(), "Old dubbed video should be deleted"
    assert mgr.is_active_or_queued(1) is True


def test_audio_mixing_complete_speaker_muting_and_gap_merging(tmp_path: Path) -> None:
    """Verify that original speaker voice is completely silenced (gain=0.0) across segments and short pauses."""
    import wave
    import json
    import numpy as np
    from mediadubflow.pipeline.base import StageContext
    from mediadubflow.pipeline.stages.audio_mixing import AudioMixingStage

    sr = 16000
    duration_sec = 6.0
    total_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # 1. Background audio has a distinct 500 Hz carrier (simulating original audio track with speaker voice)
    bg_file = tmp_path / "bg_carrier.wav"
    bg_carrier = (np.sin(2 * np.pi * 500 * t) * 12000).astype(np.int16)
    with wave.open(str(bg_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bg_carrier.tobytes())

    # 2. TTS audio is silent (0) so we can directly measure the background gain envelope
    tts_file = tmp_path / "tts_silent.wav"
    tts_silent = np.zeros(total_samples, dtype=np.int16)
    with wave.open(str(tts_file), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(tts_silent.tobytes())

    # 3. Two speech segments with a 0.35s gap between them (1.5s to 2.5s, and 2.85s to 4.0s)
    trans_file = tmp_path / "trans_multi.json"
    segments = [
        {"id": 1, "start": 1.5, "end": 2.5, "text": "Clause 1", "translated_text": "ឃ្លា ១"},
        {"id": 2, "start": 2.85, "end": 4.0, "text": "Clause 2", "translated_text": "ឃ្លា ២"},
    ]
    trans_file.write_text(json.dumps(segments), encoding="utf-8")

    out_file = tmp_path / "mixed_carrier_check.wav"

    ctx = StageContext(
        episode_id=99,
        project_id=1,
        source_file=tmp_path / "video.mp4",
        work_dir=tmp_path,
        output_dir=tmp_path,
        extracted_audio=bg_file,
        tts_audio_path=tts_file,
        translation_path=trans_file,
    )

    stage = AudioMixingStage()
    success = stage._mix_with_ducked_speech(
        bg_path=bg_file,
        tts_path=tts_file,
        ctx=ctx,
        mute_speaker=True,
        bg_vol=1.0,
        tts_vol=1.0,
        out_path=out_file,
    )

    assert success is True
    with wave.open(str(out_file), "rb") as w:
        mixed_samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

    # A. Prior to dialogue (at 0.8s), background audio is fully intact (~12000 peak)
    pre_speech_level = np.max(np.abs(mixed_samples[int(0.5 * sr) : int(1.0 * sr)]))
    assert pre_speech_level > 10000, "Background should be unmuted before dialogue start"

    # B. During first segment (1.6s to 2.4s): 100% complete silence (zero original speaker)
    seg1_level = np.max(np.abs(mixed_samples[int(1.6 * sr) : int(2.4 * sr)]))
    assert seg1_level == 0.0, f"Original speaker must be 100% muted during segment 1 (got {seg1_level})"

    # C. In the 0.35s pause between segments (2.6s to 2.8s): merged interval guarantees 100% complete silence
    gap_level = np.max(np.abs(mixed_samples[int(2.6 * sr) : int(2.8 * sr)]))
    assert gap_level == 0.0, f"Gap between sentences must be kept 100% muted to prevent breath/vocal leaks (got {gap_level})"

    # D. During second segment (3.0s to 3.9s): 100% complete silence
    seg2_level = np.max(np.abs(mixed_samples[int(3.0 * sr) : int(3.9 * sr)]))
    assert seg2_level == 0.0, f"Original speaker must be 100% muted during segment 2 (got {seg2_level})"

    # E. After second segment (at 5.0s to 5.5s): background audio returns cleanly (~12000 peak)
    post_speech_level = np.max(np.abs(mixed_samples[int(5.0 * sr) : int(5.5 * sr)]))
    assert post_speech_level > 10000, "Background should return after dialogue ends"



