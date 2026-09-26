"""
Unit tests for JobManager — queue dispatch, pipeline execution, signal emissions,
checkpoint persistence, and concurrency control.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mediadubflow.core.job_manager import JobManager
from mediadubflow.database.repository import (
    create_episode,
    create_project,
    get_episode,
)
from mediadubflow.models.orm import Base, EpisodeStatus
from mediadubflow.pipeline.base import PipelineStage, StageContext, StageResult


class DummySuccessStage(PipelineStage):
    """Dummy stage that completes successfully and emits context updates."""

    name = "Audio Extraction"

    def __init__(
        self,
        name: str = "Audio Extraction",
        key: str = "extracted_audio",
        filename: str = "audio.wav",
    ) -> None:
        self.name = name
        self.key = key
        self.filename = filename

    async def run(self, ctx: StageContext) -> StageResult:
        artifact_path = ctx.work_dir / self.filename
        artifact_path.touch()
        return StageResult(
            success=True,
            message="Dummy stage succeeded",
            context_updates={self.key: artifact_path},
        )

    def can_skip(self, ctx: StageContext) -> bool:
        return (ctx.work_dir / self.filename).exists()


class DummyFailingStage(PipelineStage):
    """Dummy stage that returns a failing StageResult."""

    name = "Dummy Failing Stage"

    async def run(self, ctx: StageContext) -> StageResult:
        return StageResult(success=False, message="Simulated stage failure")

    def can_skip(self, ctx: StageContext) -> bool:
        return False


class DummyExceptionStage(PipelineStage):
    """Dummy stage that raises an unexpected Exception."""

    name = "Dummy Exception Stage"

    async def run(self, ctx: StageContext) -> StageResult:
        raise RuntimeError("Crash in pipeline stage")

    def can_skip(self, ctx: StageContext) -> bool:
        return False


@pytest_asyncio.fixture
async def test_db() -> tuple[Any, async_sessionmaker[AsyncSession]]:
    """Create in-memory SQLite database and return session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield engine, factory
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.fixture
def patch_get_session(
    monkeypatch: pytest.MonkeyPatch, test_db: tuple[Any, async_sessionmaker[AsyncSession]]
) -> None:
    """Monkeypatch get_session across job_manager and repository to use test DB."""
    _, factory = test_db

    @asynccontextmanager
    async def _mock_get_session():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr("mediadubflow.core.job_manager.get_session", _mock_get_session)


@pytest.mark.asyncio
async def test_job_manager_enqueue_and_complete(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    """JobManager dispatches an episode, executes stages, persists checkpoints, and marks DONE."""
    _, factory = test_db
    async with factory() as session:
        project = await create_project(
            session, name="Drama", source_folder=str(tmp_path), output_folder=str(tmp_path)
        )
        source_file = tmp_path / "ep1.mp4"
        source_file.touch()
        episode = await create_episode(
            session, project_id=project.id, episode_number=1, source_file=str(source_file)
        )
        await session.commit()
        ep_id = episode.id

    # Ensure work_dir is clean so can_skip returns False
    import shutil

    from mediadubflow.config import settings

    work_dir = settings.cache_dir / f"episode_{ep_id}"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    stage1 = DummySuccessStage(name="Audio Extraction", key="extracted_audio", filename="audio.wav")
    stage2 = DummySuccessStage(
        name="Speech-to-Text", key="transcript_path", filename="transcript.json"
    )

    jm = JobManager(stages=[stage1, stage2])

    progress_events: list[tuple[int, str, int]] = []
    status_events: list[tuple[int, str]] = []
    completed_events: list[int] = []

    jm.progress_updated.connect(lambda ep, st, pct: progress_events.append((ep, st, pct)))
    jm.status_changed.connect(lambda ep, st: status_events.append((ep, st)))
    jm.episode_completed.connect(lambda ep: completed_events.append(ep))

    jm.enqueue(ep_id)

    # Run JobManager loop briefly
    worker_task = asyncio.create_task(jm.start())
    # Wait until episode is completed
    for _ in range(50):
        if completed_events:
            break
        await asyncio.sleep(0.05)

    jm.stop()
    await worker_task

    assert completed_events == [ep_id]
    assert (ep_id, EpisodeStatus.DONE.value) in status_events
    assert (ep_id, EpisodeStatus.EXTRACTING_AUDIO.value) in status_events

    # Verify checkpoints persisted in database
    async with factory() as session:
        saved = await get_episode(session, ep_id)
        assert saved is not None
        assert saved.status == EpisodeStatus.DONE
        assert saved.extracted_audio_path is not None
        assert saved.transcript_path is not None
        assert Path(saved.extracted_audio_path).name == "audio.wav"
        assert Path(saved.transcript_path).name == "transcript.json"


@pytest.mark.asyncio
async def test_job_manager_skips_stage_with_checkpoint(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    """Stages whose can_skip() returns True are skipped without error."""
    _, factory = test_db
    async with factory() as session:
        project = await create_project(
            session, name="Drama", source_folder=str(tmp_path), output_folder=str(tmp_path)
        )
        source_file = tmp_path / "ep1.mp4"
        source_file.touch()
        episode = await create_episode(
            session, project_id=project.id, episode_number=1, source_file=str(source_file)
        )
        await session.commit()
        ep_id = episode.id

    stage1 = DummySuccessStage(key="extracted_audio", filename="audio.wav")
    jm = JobManager(stages=[stage1])

    # Pre-create artifact in settings.cache_dir so stage1.can_skip() returns True
    from mediadubflow.config import settings

    work_dir = settings.cache_dir / f"episode_{ep_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    pre_existing = work_dir / "audio.wav"
    pre_existing.touch()

    completed_events: list[int] = []
    progress_events: list[tuple[int, str, int]] = []
    jm.episode_completed.connect(lambda ep: completed_events.append(ep))
    jm.progress_updated.connect(lambda ep, st, pct: progress_events.append((ep, st, pct)))

    jm.enqueue(ep_id)
    worker_task = asyncio.create_task(jm.start())

    for _ in range(50):
        if completed_events:
            break
        await asyncio.sleep(0.05)

    jm.stop()
    await worker_task

    # Pre-existing checkpoint was skipped so no progress_updated was emitted for stage1
    assert completed_events == [ep_id]
    assert len(progress_events) == 0

    # Clean up created cache dir
    if pre_existing.exists():
        pre_existing.unlink()


@pytest.mark.asyncio
async def test_job_manager_stage_failure(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    """When a stage returns success=False, episode is marked FAILED and episode_failed is emitted."""
    _, factory = test_db
    async with factory() as session:
        project = await create_project(
            session, name="Drama", source_folder=str(tmp_path), output_folder=str(tmp_path)
        )
        source_file = tmp_path / "ep1.mp4"
        source_file.touch()
        episode = await create_episode(
            session, project_id=project.id, episode_number=1, source_file=str(source_file)
        )
        await session.commit()
        ep_id = episode.id

    failing_stage = DummyFailingStage()
    jm = JobManager(stages=[failing_stage])

    failed_events: list[tuple[int, str]] = []
    jm.episode_failed.connect(lambda ep, msg: failed_events.append((ep, msg)))

    jm.enqueue(ep_id)
    worker_task = asyncio.create_task(jm.start())

    for _ in range(50):
        if failed_events:
            break
        await asyncio.sleep(0.05)

    jm.stop()
    await worker_task

    assert len(failed_events) == 1
    assert failed_events[0][0] == ep_id
    assert "Simulated stage failure" in failed_events[0][1]

    async with factory() as session:
        saved = await get_episode(session, ep_id)
        assert saved is not None
        assert saved.status == EpisodeStatus.FAILED
        assert saved.error_message == "Simulated stage failure"
        assert saved.retry_count == 1


@pytest.mark.asyncio
async def test_job_manager_stage_exception(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    """When a stage raises an exception, it is caught, marked FAILED, and signal is emitted."""
    _, factory = test_db
    async with factory() as session:
        project = await create_project(
            session, name="Drama", source_folder=str(tmp_path), output_folder=str(tmp_path)
        )
        source_file = tmp_path / "ep1.mp4"
        source_file.touch()
        episode = await create_episode(
            session, project_id=project.id, episode_number=1, source_file=str(source_file)
        )
        await session.commit()
        ep_id = episode.id

    exception_stage = DummyExceptionStage()
    jm = JobManager(stages=[exception_stage])

    failed_events: list[tuple[int, str]] = []
    jm.episode_failed.connect(lambda ep, msg: failed_events.append((ep, msg)))

    jm.enqueue(ep_id)
    worker_task = asyncio.create_task(jm.start())

    for _ in range(50):
        if failed_events:
            break
        await asyncio.sleep(0.05)

    jm.stop()
    await worker_task

    assert len(failed_events) == 1
    assert failed_events[0][0] == ep_id
    assert "Crash in pipeline stage" in failed_events[0][1]

    async with factory() as session:
        saved = await get_episode(session, ep_id)
        assert saved is not None
        assert saved.status == EpisodeStatus.FAILED
        assert "Crash in pipeline stage" in (saved.error_message or "")


@pytest.mark.asyncio
async def test_job_manager_non_existent_episode(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
) -> None:
    """Enqueueing a non-existent episode ID handles gracefully without crashing."""
    jm = JobManager(stages=[DummySuccessStage()])
    jm.enqueue(999999)

    worker_task = asyncio.create_task(jm.start())
    await asyncio.sleep(0.1)
    jm.stop()
    await worker_task


class DummyLanguageStage(PipelineStage):
    """Stage that emits language detection updates."""

    name = "Language Detection"

    async def run(self, ctx: StageContext) -> StageResult:
        return StageResult(
            success=True,
            message="Detected language zh",
            context_updates={
                "source_language": "zh",
                "language_confidence": 0.95,
            },
        )

    def can_skip(self, ctx: StageContext) -> bool:
        return bool(ctx.source_language)


@pytest.mark.asyncio
async def test_job_manager_persists_language_updates(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    """When a stage outputs source_language & language_confidence, they are saved in DB."""
    _, factory = test_db
    async with factory() as session:
        project = await create_project(
            session, name="Drama", source_folder=str(tmp_path), output_folder=str(tmp_path)
        )
        source_file = tmp_path / "ep1.mp4"
        source_file.touch()
        episode = await create_episode(
            session, project_id=project.id, episode_number=1, source_file=str(source_file)
        )
        await session.commit()
        ep_id = episode.id

    jm = JobManager(stages=[DummyLanguageStage()])
    completed_events: list[int] = []
    jm.episode_completed.connect(lambda ep: completed_events.append(ep))

    jm.enqueue(ep_id)
    worker_task = asyncio.create_task(jm.start())

    for _ in range(50):
        if completed_events:
            break
        await asyncio.sleep(0.05)

    jm.stop()
    await worker_task

    assert completed_events == [ep_id]

    async with factory() as session:
        saved = await get_episode(session, ep_id)
        assert saved is not None
        assert saved.detected_language == "zh"
        assert saved.language_confidence == pytest.approx(0.95)


class DummyGlossaryCheckStage(PipelineStage):
    """Stage that asserts glossary metadata was passed from parent project."""

    name = "Glossary Check"

    async def run(self, ctx: StageContext) -> StageResult:
        glossary = ctx.metadata.get("glossary")
        if glossary == {"King": "ស្តេច"}:
            return StageResult(success=True, message="Glossary matched")
        return StageResult(success=False, message=f"Unexpected glossary: {glossary}")

    def can_skip(self, ctx: StageContext) -> bool:
        return False


@pytest.mark.asyncio
async def test_job_manager_loads_project_glossary_into_context(
    patch_get_session: None,
    test_db: tuple[Any, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    """JobManager loads project.glossary_json into ctx.metadata['glossary']."""
    _, factory = test_db
    async with factory() as session:
        project = await create_project(
            session,
            name="Royal",
            source_folder=str(tmp_path),
            output_folder=str(tmp_path),
            glossary_json='{"King": "ស្តេច"}',
        )
        source_file = tmp_path / "ep1.mp4"
        source_file.touch()
        episode = await create_episode(
            session, project_id=project.id, episode_number=1, source_file=str(source_file)
        )
        await session.commit()
        ep_id = episode.id

    jm = JobManager(stages=[DummyGlossaryCheckStage()])
    completed_events: list[int] = []
    jm.episode_completed.connect(lambda ep: completed_events.append(ep))

    jm.enqueue(ep_id)
    worker_task = asyncio.create_task(jm.start())

    for _ in range(50):
        if completed_events:
            break
        await asyncio.sleep(0.05)

    jm.stop()
    await worker_task

    assert completed_events == [ep_id]
