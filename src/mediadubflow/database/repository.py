"""
Database repository — focused query functions for the application domain.

All direct SQLAlchemy queries are kept here so that business logic in
the job manager and GUI code never constructs queries directly.  Every
function accepts an AsyncSession and performs a single, well-named action.

Usage:
    async with get_session() as session:
        project = await create_project(session, name="My Drama", ...)
        episode = await create_episode(session, project_id=project.id, ...)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from mediadubflow.models.orm import Episode, EpisodeStatus, Project, Speaker

# ---------------------------------------------------------------------------
# Project queries
# ---------------------------------------------------------------------------


async def create_project(
    session: AsyncSession,
    *,
    name: str,
    source_folder: str | Path,
    output_folder: str | Path,
    target_language: str = "km",
    source_language: str | None = None,
    glossary_json: str | None = None,
) -> Project:
    """
    Insert a new project row and return the persisted instance.

    Args:
        session:         Active async session (caller manages commit/rollback).
        name:            Human-readable project name.
        source_folder:   Absolute path to the folder containing source episodes.
        output_folder:   Absolute path where localized output will be written.
        target_language: BCP-47 language code for the target language (default: "km").
        source_language: BCP-47 language code detected/set for source (optional).
        glossary_json:   Optional serialized JSON glossary dictionary for terminology.

    Returns:
        The newly created Project ORM instance with id populated.
    """
    project = Project(
        name=name,
        source_folder=str(source_folder),
        output_folder=str(output_folder),
        target_language=target_language,
        source_language=source_language,
        glossary_json=glossary_json,
    )
    session.add(project)
    await session.flush()  # populate id without committing yet
    logger.info("Created project id={} name={!r}", project.id, project.name)
    return project


async def update_project_glossary(
    session: AsyncSession,
    project_id: int,
    glossary_json: str | None,
) -> None:
    """
    Update or clear the terminology glossary JSON for a project.

    Args:
        session:       Active async session.
        project_id:    Project primary key.
        glossary_json: Serialized JSON dictionary of terms, or None to clear.
    """
    project = await session.get(Project, project_id)
    if project is None:
        logger.error("update_project_glossary: project id={} not found", project_id)
        return

    project.glossary_json = glossary_json
    logger.debug("Project id={} glossary updated", project_id)


async def get_project(session: AsyncSession, project_id: int) -> Project | None:
    """
    Fetch a project by primary key, eagerly loading its episodes.

    Returns None if no project with that id exists.
    """
    result = await session.execute(
        select(Project).where(Project.id == project_id).options(selectinload(Project.episodes))
    )
    return result.scalar_one_or_none()


async def list_projects(session: AsyncSession) -> list[Project]:
    """Return all projects ordered by creation date descending."""
    result = await session.execute(select(Project).order_by(Project.created_at.desc()))
    return list(result.scalars().all())


async def delete_project(session: AsyncSession, project_id: int) -> bool:
    """
    Delete a project and all its episodes (cascade).

    Returns True if a row was deleted, False if the project was not found.
    """
    project = await session.get(Project, project_id)
    if project is None:
        logger.warning("delete_project: project id={} not found", project_id)
        return False
    await session.delete(project)
    logger.info("Deleted project id={}", project_id)
    return True


# ---------------------------------------------------------------------------
# Episode queries
# ---------------------------------------------------------------------------


async def create_episode(
    session: AsyncSession,
    *,
    project_id: int,
    episode_number: int,
    source_file: str | Path,
    title: str | None = None,
) -> Episode:
    """
    Insert a new episode row linked to an existing project.

    Args:
        session:        Active async session.
        project_id:     FK reference to the parent Project.
        episode_number: Numeric episode index (extracted from filename).
        source_file:    Absolute path to the source video file.
        title:          Optional episode title.

    Returns:
        The newly created Episode ORM instance with id populated.
    """
    episode = Episode(
        project_id=project_id,
        episode_number=episode_number,
        source_file=str(source_file),
        title=title,
        status=EpisodeStatus.PENDING,
    )
    session.add(episode)
    await session.flush()
    logger.info(
        "Created episode id={} number={} project_id={}",
        episode.id,
        episode_number,
        project_id,
    )
    return episode


async def get_episode(session: AsyncSession, episode_id: int) -> Episode | None:
    """Fetch an episode by primary key, eagerly loading its parent project."""
    result = await session.execute(
        select(Episode).where(Episode.id == episode_id).options(selectinload(Episode.project))
    )
    return result.scalar_one_or_none()


async def list_episodes(session: AsyncSession, project_id: int) -> list[Episode]:
    """Return all episodes for a project ordered by episode number ascending."""
    result = await session.execute(
        select(Episode).where(Episode.project_id == project_id).order_by(Episode.episode_number)
    )
    return list(result.scalars().all())


async def update_episode_status(
    session: AsyncSession,
    episode_id: int,
    status: EpisodeStatus,
    *,
    error_message: str | None = None,
) -> None:
    """
    Update the processing status of an episode.

    Also sets started_at on the first transition out of PENDING, and
    completed_at when status is DONE or FAILED.

    Args:
        session:       Active async session.
        episode_id:    Episode to update.
        status:        New EpisodeStatus value.
        error_message: Optional error detail stored when status is FAILED.
    """
    episode = await session.get(Episode, episode_id)
    if episode is None:
        logger.error("update_episode_status: episode id={} not found", episode_id)
        return

    now = datetime.now(UTC)

    if episode.status == EpisodeStatus.PENDING and status != EpisodeStatus.PENDING:
        episode.started_at = now

    if status in (EpisodeStatus.DONE, EpisodeStatus.FAILED):
        episode.completed_at = now

    if status == EpisodeStatus.FAILED:
        episode.retry_count = (episode.retry_count or 0) + 1
        if error_message:
            episode.error_message = error_message

    episode.status = status
    logger.debug("Episode id={} status -> {}", episode_id, status.value)


async def update_episode_checkpoint(
    session: AsyncSession,
    episode_id: int,
    field: str,
    value: str | Path,
) -> None:
    """
    Persist a checkpoint path for a completed pipeline stage.

    Only the following checkpoint fields may be updated:
    extracted_audio_path, transcript_path, diarization_path,
    translation_path, subtitle_srt_path, subtitle_ass_path,
    tts_audio_path, mixed_audio_path, output_video_path.

    Args:
        session:    Active async session.
        episode_id: Episode to update.
        field:      Name of the checkpoint column on Episode.
        value:      Absolute path to the produced artifact.

    Raises:
        ValueError: If field is not a recognised checkpoint column.
    """
    allowed_fields = {
        "extracted_audio_path",
        "transcript_path",
        "diarization_path",
        "translation_path",
        "subtitle_srt_path",
        "subtitle_ass_path",
        "tts_audio_path",
        "mixed_audio_path",
        "output_video_path",
    }
    if field not in allowed_fields:
        raise ValueError(
            f"update_episode_checkpoint: {field!r} is not a valid checkpoint field. "
            f"Allowed: {sorted(allowed_fields)}"
        )

    episode = await session.get(Episode, episode_id)
    if episode is None:
        logger.error("update_episode_checkpoint: episode id={} not found", episode_id)
        return

    setattr(episode, field, str(value))
    logger.debug("Episode id={} checkpoint {}={}", episode_id, field, value)


async def update_episode_language(
    session: AsyncSession,
    episode_id: int,
    language: str,
    confidence: float | None = None,
) -> None:
    """
    Persist detected source language and detection confidence to an episode.

    Args:
        session:    Active async session.
        episode_id: Episode primary key.
        language:   ISO 639-1 language code (e.g. 'zh', 'en', 'ko').
        confidence: Detection probability between 0.0 and 1.0.
    """
    episode = await session.get(Episode, episode_id)
    if episode is None:
        logger.error("update_episode_language: episode id={} not found", episode_id)
        return

    episode.detected_language = language
    if confidence is not None:
        episode.language_confidence = confidence
    logger.debug(
        "Episode id={} language -> {} (confidence={})",
        episode_id,
        language,
        confidence,
    )


# ---------------------------------------------------------------------------
# Speaker queries
# ---------------------------------------------------------------------------


async def upsert_speaker(
    session: AsyncSession,
    *,
    episode_id: int,
    label: str,
    display_name: str | None = None,
    tts_voice_id: str | None = None,
) -> Speaker:
    """
    Insert or update a speaker row for the given episode + label.

    If a Speaker with (episode_id, label) already exists it is updated
    in-place; otherwise a new row is inserted.

    Args:
        session:      Active async session.
        episode_id:   FK reference to the parent Episode.
        label:        Diarization label e.g. "SPEAKER_00".
        display_name: Human-readable name assigned in the review UI.
        tts_voice_id: TTS voice identifier to use for this speaker.

    Returns:
        The inserted or updated Speaker instance.
    """
    result = await session.execute(
        select(Speaker).where(
            Speaker.episode_id == episode_id,
            Speaker.label == label,
        )
    )
    speaker = result.scalar_one_or_none()

    if speaker is None:
        speaker = Speaker(episode_id=episode_id, label=label)
        session.add(speaker)

    if display_name is not None:
        speaker.display_name = display_name
    if tts_voice_id is not None:
        speaker.tts_voice_id = tts_voice_id

    await session.flush()
    return speaker
