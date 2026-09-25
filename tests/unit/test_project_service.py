"""
Unit tests for the ProjectService application service.

Tests folder scanning, atomic project and episode creation, path sanitization,
and incremental episode re-scanning using an in-memory SQLite database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mediadubflow.config.settings import settings
from mediadubflow.database.repository import list_episodes
from mediadubflow.models.orm import Base, EpisodeStatus
from mediadubflow.services.project_service import (
    _sanitize_folder_name,
    create_project_from_folder,
    rescan_project_episodes,
)


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    """Provide an isolated in-memory SQLite session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def test_sanitize_folder_name() -> None:
    """Verify illegal filesystem characters and spaces are sanitized."""
    assert (
        _sanitize_folder_name('Drama: Season 1 / Ep "Special" *?')
        == "Drama__Season_1___Ep__Special"
    )
    assert _sanitize_folder_name("   ") == "project"
    assert _sanitize_folder_name("...test...") == "test"
    assert _sanitize_folder_name("My Drama") == "My_Drama"


@pytest.mark.asyncio
async def test_create_project_from_folder_success(session: AsyncSession, tmp_path: Path) -> None:
    """Successfully scan folder and persist project with all detected episodes."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "Drama_E01.mp4").touch()
    (source_dir / "Drama_E02.mkv").touch()
    (source_dir / "Drama_E03.mp4").touch()
    (source_dir / "notes.txt").touch()  # Non-video file should be ignored

    project = await create_project_from_folder(
        session,
        name="Royal Romance",
        source_folder=source_dir,
    )
    await session.commit()

    assert project.id is not None
    assert project.name == "Royal Romance"
    assert project.source_folder == str(source_dir.resolve())
    assert project.target_language == "km"
    assert len(project.episodes) == 3

    episodes = await list_episodes(session, project.id)
    assert len(episodes) == 3
    numbers = [ep.episode_number for ep in episodes]
    assert numbers == [1, 2, 3]
    assert all(ep.status == EpisodeStatus.PENDING for ep in episodes)


@pytest.mark.asyncio
async def test_create_project_from_folder_default_output(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Omitting output_folder uses settings.output_root / sanitized_name."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "ep1.mp4").touch()

    project = await create_project_from_folder(
        session,
        name="Drama 2026: Part 1",
        source_folder=source_dir,
    )
    await session.commit()

    expected_output = (settings.output_root / "Drama_2026__Part_1").resolve()
    assert project.output_folder == str(expected_output)


@pytest.mark.asyncio
async def test_create_project_from_folder_custom_output(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Supplying output_folder preserves the custom path."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "ep1.mp4").touch()

    custom_out = tmp_path / "custom_output"

    project = await create_project_from_folder(
        session,
        name="Drama Custom",
        source_folder=source_dir,
        output_folder=custom_out,
    )
    await session.commit()

    assert project.output_folder == str(custom_out.resolve())


@pytest.mark.asyncio
async def test_create_project_from_folder_empty_folder(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Scanning an empty folder or a folder with no videos raises ValueError."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").touch()

    with pytest.raises(ValueError, match="No supported video files found"):
        await create_project_from_folder(
            session,
            name="Empty Drama",
            source_folder=empty_dir,
        )


@pytest.mark.asyncio
async def test_create_project_from_folder_nonexistent_path(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Non-existent source folder raises FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        await create_project_from_folder(
            session,
            name="Missing Drama",
            source_folder=nonexistent,
        )


@pytest.mark.asyncio
async def test_create_project_from_folder_empty_name(session: AsyncSession, tmp_path: Path) -> None:
    """Empty or blank project name raises ValueError."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "ep1.mp4").touch()

    with pytest.raises(ValueError, match="Project name cannot be empty"):
        await create_project_from_folder(
            session,
            name="   ",
            source_folder=source_dir,
        )


@pytest.mark.asyncio
async def test_rescan_project_episodes_adds_new_files(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Re-scanning adds only new episodes without duplicating existing ones."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "ep1.mp4").touch()

    project = await create_project_from_folder(
        session,
        name="Serial Show",
        source_folder=source_dir,
    )
    await session.commit()
    assert len(project.episodes) == 1

    # Add 2 more episodes to the directory
    (source_dir / "ep2.mp4").touch()
    (source_dir / "ep3.mp4").touch()

    new_episodes = await rescan_project_episodes(session, project.id)
    await session.commit()

    assert len(new_episodes) == 2
    assert [ep.episode_number for ep in new_episodes] == [2, 3]

    all_episodes = await list_episodes(session, project.id)
    assert len(all_episodes) == 3


@pytest.mark.asyncio
async def test_rescan_project_episodes_no_new_files(session: AsyncSession, tmp_path: Path) -> None:
    """Re-scanning when no new files exist returns empty list."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "ep1.mp4").touch()

    project = await create_project_from_folder(
        session,
        name="Show",
        source_folder=source_dir,
    )
    await session.commit()

    new_episodes = await rescan_project_episodes(session, project.id)
    assert new_episodes == []


@pytest.mark.asyncio
async def test_rescan_project_not_found(session: AsyncSession) -> None:
    """Re-scanning a non-existent project id raises ValueError."""
    with pytest.raises(ValueError, match="Project with id 9999 not found"):
        await rescan_project_episodes(session, 9999)


@pytest.mark.asyncio
async def test_create_project_with_glossary(session: AsyncSession, tmp_path: Path) -> None:
    """Project creation persists custom terminology glossary."""
    source_dir = tmp_path / "drama_source"
    source_dir.mkdir()
    (source_dir / "ep1.mp4").touch()

    project = await create_project_from_folder(
        session,
        name="Historical Romance",
        source_folder=source_dir,
        glossary_json='{"Emperor": "ព្រះចៅអធិរាជ"}',
    )
    await session.commit()

    assert project.glossary_json == '{"Emperor": "ព្រះចៅអធិរាជ"}'
