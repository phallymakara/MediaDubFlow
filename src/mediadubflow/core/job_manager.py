"""
Job Manager — orchestrates episode processing through the pipeline.

Responsibilities:
- Maintain an asyncio-based episode queue.
- Enforce the ``max_concurrent_episodes`` concurrency limit.
- Run each stage in sequence, skipping stages whose checkpoint exists.
- Persist episode status updates to the database after each stage.
- Emit Qt signals so the GUI can update progress without polling.
- Handle per-episode errors without stopping the entire project.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from PySide6.QtCore import QObject, Signal

from mediadubflow.config import settings
from mediadubflow.database.repository import (
    get_episode,
    update_episode_checkpoint,
    update_episode_language,
    update_episode_status,
)
from mediadubflow.database.session import get_session
from mediadubflow.models.orm import EpisodeStatus
from mediadubflow.pipeline.base import PipelineStage, StageContext
from mediadubflow.pipeline.stages.audio_extraction import AudioExtractionStage
from mediadubflow.pipeline.stages.audio_mixing import AudioMixingStage
from mediadubflow.pipeline.stages.language_detection import LanguageDetectionStage
from mediadubflow.pipeline.stages.speaker_detection import SpeakerDetectionStage
from mediadubflow.pipeline.stages.subtitle_generation import SubtitleGenerationStage
from mediadubflow.pipeline.stages.transcription import TranscriptionStage
from mediadubflow.pipeline.stages.translation import TranslationStage
from mediadubflow.pipeline.stages.tts_generation import TTSGenerationStage
from mediadubflow.pipeline.stages.video_rendering import VideoRenderingStage

# Ordered pipeline stages:
# Audio must be extracted first so language detection and transcription can run on 16kHz WAV.
PIPELINE: list[PipelineStage] = [
    AudioExtractionStage(),
    LanguageDetectionStage(),
    TranscriptionStage(),
    SpeakerDetectionStage(),
    TranslationStage(),
    SubtitleGenerationStage(),
    TTSGenerationStage(),
    AudioMixingStage(),
    VideoRenderingStage(),
]

# Mapping between StageResult.context_updates keys and Episode checkpoint column names
CONTEXT_CHECKPOINT_MAP: dict[str, str] = {
    "extracted_audio": "extracted_audio_path",
    "transcript_path": "transcript_path",
    "diarization_path": "diarization_path",
    "translation_path": "translation_path",
    "subtitle_srt_path": "subtitle_srt_path",
    "subtitle_ass_path": "subtitle_ass_path",
    "tts_audio_path": "tts_audio_path",
    "mixed_audio_path": "mixed_audio_path",
    "output_video_path": "output_video_path",
}

# Mapping between stage names and EpisodeStatus enum values
STAGE_STATUS_MAP: dict[str, EpisodeStatus] = {
    "Audio Extraction": EpisodeStatus.EXTRACTING_AUDIO,
    "Language Detection": EpisodeStatus.DETECTING_LANGUAGE,
    "Speech-to-Text": EpisodeStatus.TRANSCRIBING,
    "Speaker Detection": EpisodeStatus.DETECTING_SPEAKERS,
    "Translation": EpisodeStatus.TRANSLATING,
    "Subtitle Generation": EpisodeStatus.GENERATING_SUBTITLES,
    "TTS Generation": EpisodeStatus.GENERATING_TTS,
    "Audio Mixing": EpisodeStatus.MIXING_AUDIO,
    "Video Rendering": EpisodeStatus.RENDERING,
}


class JobManager(QObject):
    """
    Async job manager that drives the episode processing pipeline.

    Emits Qt signals consumed by the GUI for live progress updates.
    Should be created once and run in a dedicated asyncio event loop
    (via ``QThread + asyncio.run``).
    """

    # episode_id, stage_name, progress_pct (0-100)
    progress_updated: Signal = Signal(int, str, int)
    # episode_id, EpisodeStatus value
    status_changed: Signal = Signal(int, str)
    # episode_id, error message
    episode_failed: Signal = Signal(int, str)
    # episode_id
    episode_completed: Signal = Signal(int)

    def __init__(
        self,
        parent: QObject | None = None,
        stages: list[PipelineStage] | None = None,
    ) -> None:
        super().__init__(parent)
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._active_or_queued: set[int] = set()
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._running = False
        self._stages = list(stages) if stages is not None else list(PIPELINE)

    async def start(self) -> None:
        """Start the worker loop. Call once from the async entry point."""
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_episodes)
        self._running = True
        logger.info(
            "JobManager started — max_concurrent_episodes={}",
            settings.max_concurrent_episodes,
        )
        await self._dispatch_loop()

    def stop(self, cancel_active: bool = False) -> None:
        """Signal the worker loop to stop after finishing current jobs, or cancel immediately."""
        self._running = False
        if cancel_active:
            for task in list(self._tasks.values()):
                if not task.done():
                    task.cancel()

    def cancel_episode(self, episode_id: int) -> bool:
        """Cancel a running episode processing task."""
        task = self._tasks.get(episode_id)
        if task and not task.done():
            task.cancel()
            logger.info("[Episode {}] Cancelled active episode task", episode_id)
            return True
        return False

    def is_active_or_queued(self, episode_id: int) -> bool:
        """Check if an episode is already in the queue or actively running."""
        return episode_id in self._active_or_queued

    def enqueue(self, episode_id: int) -> bool:
        """Add an episode to the processing queue if not already queued or running."""
        if episode_id in self._active_or_queued:
            logger.warning("[Episode {}] Already queued or running; ignoring duplicate enqueue", episode_id)
            return False
        self._active_or_queued.add(episode_id)
        self._queue.put_nowait(episode_id)
        logger.info("Enqueued episode_id={}", episode_id)
        return True

    def redub_episode(self, episode_id: int, new_speed: float | None = None) -> bool:
        """
        Invalidate downstream audio and video checkpoints for an already dubbed episode,
        apply the new speech speed, and enqueue for rapid re-dubbing.
        """
        if new_speed is not None:
            settings.tts_speed = round(float(new_speed), 2)
            logger.info("[Episode {}] Updated TTS speed to {:.2f}x for re-dubbing", episode_id, settings.tts_speed)

        # 1. Clean up audio and video artifacts from work_dir and output_dir
        work_dir = settings.cache_dir / f"episode_{episode_id}"
        output_dir = settings.output_root / f"episode_{episode_id}"

        for p in (
            work_dir / f"dialogue_{episode_id}.wav",
            work_dir / f"mixed_{episode_id}.wav",
        ):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

        chunks_dir = work_dir / "tts_chunks"
        if chunks_dir.exists():
            import shutil  # noqa: PLC0415

            try:
                shutil.rmtree(chunks_dir, ignore_errors=True)
            except OSError:
                pass

        # Also remove existing output video so VideoRendering re-muxes with new audio
        if output_dir.exists():
            for vid in output_dir.glob("*_dubbed.mp4"):
                try:
                    vid.unlink()
                except OSError:
                    pass

        # 2. Reset database checkpoints in background if loop is active
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._reset_checkpoints_for_redub(episode_id))
        except RuntimeError:
            pass

        # 3. Enqueue episode (clear active status if it was stuck as completed)
        self._active_or_queued.discard(episode_id)
        return self.enqueue(episode_id)

    async def _reset_checkpoints_for_redub(self, episode_id: int) -> None:
        """Reset downstream database checkpoint paths and set status to QUEUED."""
        async with get_session() as session:
            await update_episode_checkpoint(session, episode_id=episode_id, field="tts_audio_path", value="")
            await update_episode_checkpoint(session, episode_id=episode_id, field="mixed_audio_path", value="")
            await update_episode_checkpoint(session, episode_id=episode_id, field="output_video_path", value="")
            await update_episode_status(session, episode_id=episode_id, status=EpisodeStatus.QUEUED)

    async def _dispatch_loop(self) -> None:
        """Pull episodes from the queue and run them concurrently."""
        tasks: set[asyncio.Task[None]] = set()

        while self._running:
            try:
                episode_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            assert self._semaphore is not None
            task = asyncio.create_task(
                self._run_episode(episode_id, self._semaphore),
                name=f"episode-{episode_id}",
            )
            tasks.add(task)
            self._tasks[episode_id] = task
            task.add_done_callback(tasks.discard)
            task.add_done_callback(lambda _, ep_id=episode_id: self._tasks.pop(ep_id, None))

        # Wait for all in-flight tasks before returning.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_episode(self, episode_id: int, sem: asyncio.Semaphore) -> None:
        """Execute the full pipeline for a single episode under the semaphore."""
        try:
            async with sem:
                logger.info("Processing episode_id={}", episode_id)
                try:
                    ctx = await self._build_context(episode_id)
                except Exception as exc:
                    logger.exception("[Episode {}] Failed to build context: {}", episode_id, exc)
                    safe_msg = f"Initialization error: {type(exc).__name__}"
                    await self._set_failed(episode_id, safe_msg)
                    self.episode_failed.emit(episode_id, safe_msg)
                    return

                if ctx is None:
                    safe_msg = "Invalid or missing episode source file"
                    await self._set_failed(episode_id, safe_msg)
                    self.episode_failed.emit(episode_id, safe_msg)
                    return

                total_stages = max(1, len(self._stages))
                for stage_idx, stage in enumerate(self._stages):
                    if stage.can_skip(ctx):
                        logger.debug(
                            "[Episode {}] Skipping stage '{}' (checkpoint found)",
                            episode_id,
                            stage.name,
                        )
                        skip_pct = int(((stage_idx + 1) / total_stages) * 100)
                        self.progress_updated.emit(
                            episode_id,
                            f"Step {stage_idx + 1}/{total_stages}: {stage.name} (Cached)",
                            skip_pct,
                        )
                        continue

                    stage_status = STAGE_STATUS_MAP.get(stage.name, EpisodeStatus.QUEUED)
                    await self._set_status(episode_id, stage_status)

                    def _on_stage_progress(intra_pct: int, s_idx: int = stage_idx, s_name: str = stage.name) -> None:
                        clamped_intra = max(0, min(100, intra_pct))
                        overall_pct = int(((s_idx + (clamped_intra / 100.0)) / total_stages) * 100)
                        stage_desc = f"Step {s_idx + 1}/{total_stages}: {s_name}"
                        self.progress_updated.emit(episode_id, stage_desc, overall_pct)

                    ctx.progress_callback = _on_stage_progress
                    _on_stage_progress(0)

                    try:
                        result = await stage.run(ctx)
                    except asyncio.CancelledError:
                        logger.info("[Episode {}] Stage '{}' was cancelled", episode_id, stage.name)
                        safe_msg = f"{stage.name} cancelled by user"
                        await self._set_failed(episode_id, safe_msg)
                        self.episode_failed.emit(episode_id, safe_msg)
                        raise
                    except Exception as exc:
                        logger.exception(
                            "[Episode {}] Stage '{}' raised an exception: {}",
                            episode_id,
                            stage.name,
                            exc,
                        )
                        safe_msg = f"{stage.name} failed: {exc}"
                        await self._set_failed(episode_id, safe_msg)
                        self.episode_failed.emit(episode_id, safe_msg)
                        return

                    if not result.success:
                        await self._set_failed(episode_id, result.message)
                        self.episode_failed.emit(episode_id, result.message)
                        return

                    # Apply context updates returned by the stage and persist checkpoints to DB.
                    await self._persist_stage_updates(episode_id, ctx, result.context_updates)

                    _on_stage_progress(100)

                self.progress_updated.emit(episode_id, "Complete", 100)
                await self._set_status(episode_id, EpisodeStatus.DONE)
                self.episode_completed.emit(episode_id)
                logger.info("Episode {} completed successfully", episode_id)
        except Exception as exc:
            logger.exception("[Episode {}] Unexpected pipeline error: {}", episode_id, exc)
            safe_msg = f"Pipeline error: {type(exc).__name__}"
            await self._set_failed(episode_id, safe_msg)
            self.episode_failed.emit(episode_id, safe_msg)
        finally:
            self._active_or_queued.discard(episode_id)

    async def _persist_stage_updates(
        self,
        episode_id: int,
        ctx: StageContext,
        updates: dict[str, object],
    ) -> None:
        """Apply stage updates to the StageContext and persist to the database."""
        async with get_session() as session:
            for key, value in updates.items():
                if hasattr(ctx, key):
                    setattr(ctx, key, value)

                if key in CONTEXT_CHECKPOINT_MAP and value is not None:
                    db_field = CONTEXT_CHECKPOINT_MAP[key]
                    await update_episode_checkpoint(
                        session,
                        episode_id=episode_id,
                        field=db_field,
                        value=value,
                    )

            # Persist language detection updates if present
            if "source_language" in updates and updates["source_language"]:
                lang = str(updates["source_language"])
                conf = (
                    float(updates["language_confidence"])
                    if "language_confidence" in updates
                    and updates["language_confidence"] is not None
                    else None
                )
                await update_episode_language(
                    session,
                    episode_id=episode_id,
                    language=lang,
                    confidence=conf,
                )

    async def _build_context(self, episode_id: int) -> StageContext | None:
        """Load episode from DB via repository and build initial StageContext."""
        async with get_session() as session:
            episode = await get_episode(session, episode_id)
            if episode is None:
                logger.error("Episode {} not found in database", episode_id)
                return None

            work_dir = settings.cache_dir / f"episode_{episode_id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            output_dir = settings.output_root / f"episode_{episode_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            metadata: dict[str, object] = {}
            if episode.project and episode.project.glossary_json:
                try:
                    import json  # noqa: PLC0415

                    metadata["glossary"] = json.loads(episode.project.glossary_json)
                except Exception as exc:
                    logger.warning(
                        "[Episode {}] Could not parse project glossary: {}", episode_id, exc
                    )

            source_path = Path(episode.source_file).resolve()
            if not source_path.exists() or not source_path.is_file():
                logger.error(
                    "[Episode {}] Source file does not exist or is invalid: {}",
                    episode_id,
                    source_path,
                )
                return None

            allowed_dirs = (
                work_dir.resolve(),
                output_dir.resolve(),
                settings.output_root.resolve(),
            )

            def _safe_checkpoint(raw: str | None) -> Path | None:
                if not raw:
                    return None
                try:
                    p = Path(raw).resolve()
                    for allowed in allowed_dirs:
                        if p == allowed or p.is_relative_to(allowed):
                            return p
                    logger.warning(
                        "[Episode {}] Checkpoint path '{}' is outside allowed directories; ignoring",
                        episode_id,
                        raw,
                    )
                    return None
                except (ValueError, RuntimeError):
                    return None

            return StageContext(
                episode_id=episode_id,
                project_id=episode.project_id,
                source_file=source_path,
                work_dir=work_dir,
                output_dir=output_dir,
                source_language=episode.detected_language or "",
                extracted_audio=_safe_checkpoint(episode.extracted_audio_path),
                transcript_path=_safe_checkpoint(episode.transcript_path),
                diarization_path=_safe_checkpoint(episode.diarization_path),
                translation_path=_safe_checkpoint(episode.translation_path),
                subtitle_srt_path=_safe_checkpoint(episode.subtitle_srt_path),
                subtitle_ass_path=_safe_checkpoint(episode.subtitle_ass_path),
                tts_audio_path=_safe_checkpoint(episode.tts_audio_path),
                mixed_audio_path=_safe_checkpoint(episode.mixed_audio_path),
                output_video_path=_safe_checkpoint(episode.output_video_path),
                metadata=metadata,
            )

    async def _set_status(self, episode_id: int, status: EpisodeStatus) -> None:
        """Update episode status via repository layer and emit Qt signal."""
        async with get_session() as session:
            await update_episode_status(session, episode_id=episode_id, status=status)
        self.status_changed.emit(episode_id, status.value)

    async def _set_failed(self, episode_id: int, message: str) -> None:
        """Mark episode failed via repository layer and emit Qt signal."""
        async with get_session() as session:
            await update_episode_status(
                session,
                episode_id=episode_id,
                status=EpisodeStatus.FAILED,
                error_message=message,
            )
        self.status_changed.emit(episode_id, EpisodeStatus.FAILED.value)
