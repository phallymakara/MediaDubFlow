"""
Unit tests for database session and repository layer.

Each test creates an isolated in-memory SQLite database so that the real
mediadubflow.db is never touched.  The in-memory DB is torn down after
every test automatically.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mediadubflow.database.repository import (
    create_episode,
    create_project,
    delete_project,
    get_episode,
    get_project,
    list_episodes,
    list_projects,
    update_episode_checkpoint,
    update_episode_status,
    upsert_speaker,
)
from mediadubflow.models.orm import Base, EpisodeStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    """
    Provide a fresh in-memory SQLite async session for each test.

    Creates the full schema via create_all (no Alembic needed for tests)
    and drops it afterwards.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Project tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_project(session: AsyncSession) -> None:
    """A created project is retrievable with all fields intact."""
    project = await create_project(
        session,
        name="Test Drama",
        source_folder="/videos/drama",
        output_folder="/output/drama",
        target_language="km",
        source_language="zh",
        glossary_json='{"Xiao Ming": "សៀវ មីង"}',
    )
    await session.commit()

    fetched = await get_project(session, project.id)
    assert fetched is not None
    assert fetched.name == "Test Drama"
    assert fetched.source_folder == "/videos/drama"
    assert fetched.output_folder == "/output/drama"
    assert fetched.target_language == "km"
    assert fetched.source_language == "zh"
    assert fetched.glossary_json == '{"Xiao Ming": "សៀវ មីង"}'


@pytest.mark.asyncio
async def test_update_project_glossary(session: AsyncSession) -> None:
    """update_project_glossary updates the glossary JSON on an existing project."""
    from mediadubflow.database.repository import update_project_glossary

    project = await create_project(
        session,
        name="Test Drama",
        source_folder="/videos/drama",
        output_folder="/output/drama",
    )
    await session.commit()

    await update_project_glossary(session, project.id, '{"General": "មេទ័ព"}')
    await session.commit()

    fetched = await get_project(session, project.id)
    assert fetched is not None
    assert fetched.glossary_json == '{"General": "មេទ័ព"}'


@pytest.mark.asyncio
async def test_list_projects(session: AsyncSession) -> None:
    """list_projects returns all inserted projects."""
    await create_project(session, name="Drama A", source_folder="/a", output_folder="/oa")
    await create_project(session, name="Drama B", source_folder="/b", output_folder="/ob")
    await session.commit()

    projects = await list_projects(session)
    assert len(projects) == 2
    names = {p.name for p in projects}
    assert names == {"Drama A", "Drama B"}


@pytest.mark.asyncio
async def test_get_project_not_found(session: AsyncSession) -> None:
    """get_project returns None for a non-existent id."""
    result = await get_project(session, 9999)
    assert result is None


@pytest.mark.asyncio
async def test_delete_project_cascades(session: AsyncSession) -> None:
    """Deleting a project removes its episodes (cascade)."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session,
        project_id=project.id,
        episode_number=1,
        source_file="/src/ep1.mp4",
    )
    await session.commit()

    deleted = await delete_project(session, project.id)
    await session.commit()

    assert deleted is True
    assert await get_project(session, project.id) is None
    assert await get_episode(session, episode.id) is None


# ---------------------------------------------------------------------------
# Episode tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_episode(session: AsyncSession) -> None:
    """A created episode is linked to its project and starts as PENDING."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session,
        project_id=project.id,
        episode_number=3,
        source_file="/src/ep3.mkv",
        title="Episode 3",
    )
    await session.commit()

    fetched = await get_episode(session, episode.id)
    assert fetched is not None
    assert fetched.project_id == project.id
    assert fetched.episode_number == 3
    assert fetched.source_file == "/src/ep3.mkv"
    assert fetched.title == "Episode 3"
    assert fetched.status == EpisodeStatus.PENDING


@pytest.mark.asyncio
async def test_list_episodes_ordered(session: AsyncSession) -> None:
    """list_episodes returns episodes sorted by episode_number ascending."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    await create_episode(session, project_id=project.id, episode_number=3, source_file="/ep3.mp4")
    await create_episode(session, project_id=project.id, episode_number=1, source_file="/ep1.mp4")
    await create_episode(session, project_id=project.id, episode_number=2, source_file="/ep2.mp4")
    await session.commit()

    episodes = await list_episodes(session, project.id)
    assert [e.episode_number for e in episodes] == [1, 2, 3]


@pytest.mark.asyncio
async def test_update_episode_status(session: AsyncSession) -> None:
    """Updating episode status persists correctly and sets started_at."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    await update_episode_status(session, episode.id, EpisodeStatus.TRANSCRIBING)
    await session.commit()

    fetched = await get_episode(session, episode.id)
    assert fetched is not None
    assert fetched.status == EpisodeStatus.TRANSCRIBING
    assert fetched.started_at is not None


@pytest.mark.asyncio
async def test_update_episode_status_failed_increments_retry(session: AsyncSession) -> None:
    """Transitioning to FAILED increments retry_count and stores error_message."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    await update_episode_status(
        session, episode.id, EpisodeStatus.FAILED, error_message="Something went wrong"
    )
    await session.commit()

    fetched = await get_episode(session, episode.id)
    assert fetched is not None
    assert fetched.status == EpisodeStatus.FAILED
    assert fetched.retry_count == 1
    assert fetched.error_message == "Something went wrong"
    assert fetched.completed_at is not None


@pytest.mark.asyncio
async def test_update_episode_checkpoint(session: AsyncSession) -> None:
    """Checkpoint path is persisted for a valid field name."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    await update_episode_checkpoint(
        session, episode.id, "transcript_path", "/cache/ep1/transcript.json"
    )
    await session.commit()

    fetched = await get_episode(session, episode.id)
    assert fetched is not None
    assert fetched.transcript_path == "/cache/ep1/transcript.json"


@pytest.mark.asyncio
async def test_update_episode_checkpoint_invalid_field(session: AsyncSession) -> None:
    """update_episode_checkpoint raises ValueError for unknown field names."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    with pytest.raises(ValueError, match="not a valid checkpoint field"):
        await update_episode_checkpoint(session, episode.id, "invalid_field", "/some/path")


# ---------------------------------------------------------------------------
# Speaker tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_speaker_insert(session: AsyncSession) -> None:
    """upsert_speaker creates a new speaker when none exists."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    speaker = await upsert_speaker(
        session,
        episode_id=episode.id,
        label="SPEAKER_00",
        display_name="Hero",
    )
    await session.commit()

    assert speaker.id is not None
    assert speaker.label == "SPEAKER_00"
    assert speaker.display_name == "Hero"


@pytest.mark.asyncio
async def test_upsert_speaker_update(session: AsyncSession) -> None:
    """upsert_speaker updates an existing speaker without creating a duplicate."""
    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    speaker1 = await upsert_speaker(session, episode_id=episode.id, label="SPEAKER_00")
    await session.commit()

    speaker2 = await upsert_speaker(
        session,
        episode_id=episode.id,
        label="SPEAKER_00",
        tts_voice_id="km_voice_01",
    )
    await session.commit()

    # Same row updated, not a new row.
    assert speaker1.id == speaker2.id
    assert speaker2.tts_voice_id == "km_voice_01"


@pytest.mark.asyncio
async def test_update_episode_language(session: AsyncSession) -> None:
    """update_episode_language persists detected language code and confidence."""
    from mediadubflow.database.repository import update_episode_language

    project = await create_project(
        session, name="Drama", source_folder="/src", output_folder="/out"
    )
    episode = await create_episode(
        session, project_id=project.id, episode_number=1, source_file="/ep1.mp4"
    )
    await session.commit()

    await update_episode_language(session, episode.id, "zh", confidence=0.985)
    await session.commit()

    updated = await get_episode(session, episode.id)
    assert updated is not None
    assert updated.detected_language == "zh"
    assert updated.language_confidence == pytest.approx(0.985)
