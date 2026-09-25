"""
Project management application service.

Coordinates between filesystem episode scanning and the database repository.
Ensures project creation and batch episode population run as a single atomic transaction.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from mediadubflow.config.settings import settings
from mediadubflow.database.repository import (
    create_episode,
    create_project,
    delete_project,
    get_project,
)
from mediadubflow.models.orm import Episode, Project
from mediadubflow.utils.episode_detector import detect_episodes

# Characters illegal on Windows / POSIX file paths
_ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def _sanitize_folder_name(name: str) -> str:
    """
    Sanitize a project title to make it safe for directory naming.

    Strips illegal characters, replaces spaces with underscores, and trims whitespace.
    """
    sanitized = _ILLEGAL_PATH_CHARS.sub("_", name)
    sanitized = re.sub(r"\s+", "_", sanitized).strip(" ._")
    return sanitized if sanitized else "project"


async def create_project_from_folder(
    session: AsyncSession,
    *,
    name: str,
    source_folder: str | Path,
    output_folder: str | Path | None = None,
    target_language: str = "km",
    source_language: str | None = None,
    glossary_json: str | None = None,
    episodes: list[tuple[int, Path]] | None = None,
) -> Project:
    """
    Create a new project by scanning a drama directory for video files.

    Performs validation, detects episodes, and saves both the Project record
    and all detected Episode records within the current database transaction.

    Args:
        session: Active database AsyncSession.
        name: Human-readable project/drama title.
        source_folder: Path to directory containing video files.
        output_folder: Path to destination output directory. Defaults to
                       settings.output_root / sanitized_project_name.
        target_language: Target language code (default "km" for Khmer).
        source_language: Optional source language code (e.g. "zh", "ko", "en").
        glossary_json: Optional serialized JSON dictionary of terms for translation.

    Returns:
        Persisted Project instance with its episodes collection populated.

    Raises:
        ValueError: If name is empty or no video files are detected.
        FileNotFoundError: If source_folder does not exist.
        NotADirectoryError: If source_folder is not a directory.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Project name cannot be empty.")

    source_path = Path(source_folder).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_path}")

    # Determine destination directory
    if output_folder is None:
        dest_folder = (settings.output_root / _sanitize_folder_name(clean_name)).resolve()
    else:
        dest_folder = Path(output_folder).resolve()

    # Detect video files or use filtered selection
    detected = episodes if episodes is not None else detect_episodes(source_path)
    if not detected:
        raise ValueError(f"No supported video files found or selected in: {source_path}")

    logger.info(
        "Creating project '{!r}' from folder '{}' with {} detected episodes",
        clean_name,
        source_path,
        len(detected),
    )

    project = await create_project(
        session,
        name=clean_name,
        source_folder=str(source_path),
        output_folder=str(dest_folder),
        target_language=target_language,
        source_language=source_language,
        glossary_json=glossary_json,
    )

    for ep_num, file_path in detected:
        await create_episode(
            session,
            project_id=project.id,
            episode_number=ep_num,
            source_file=str(file_path.resolve()),
        )

    await session.flush()

    # Reload project with episodes relationship populated
    reloaded = await get_project(session, project.id)
    return reloaded or project


async def rescan_project_episodes(
    session: AsyncSession,
    project_id: int,
) -> list[Episode]:
    """
    Scan a project's source directory for any newly added video files.

    Inserts newly discovered episodes into the database without altering
    or resetting existing episodes.

    Args:
        session: Active database AsyncSession.
        project_id: Primary key of the project to re-scan.

    Returns:
        List of newly created Episode records (empty if no new files found).

    Raises:
        ValueError: If project_id is not found in the database.
    """
    project = await get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project with id {project_id} not found.")

    source_path = Path(project.source_folder)
    detected = detect_episodes(source_path)

    existing_files = {Path(ep.source_file).resolve() for ep in project.episodes}
    existing_numbers = {ep.episode_number for ep in project.episodes}

    new_episodes: list[Episode] = []
    for ep_num, file_path in detected:
        resolved_path = file_path.resolve()
        if resolved_path in existing_files:
            continue

        # If detected episode number collides with an existing one, find next available
        target_num = ep_num
        while target_num in existing_numbers:
            target_num += 1

        existing_numbers.add(target_num)
        existing_files.add(resolved_path)

        new_ep = await create_episode(
            session,
            project_id=project.id,
            episode_number=target_num,
            source_file=str(resolved_path),
        )
        new_episodes.append(new_ep)

    if new_episodes:
        await session.flush()
        logger.info(
            "Re-scan for project id={}: added {} new episodes",
            project_id,
            len(new_episodes),
        )
    else:
        logger.info("Re-scan for project id={}: no new episodes found", project_id)

    return new_episodes


async def delete_project_with_files(session: AsyncSession, project_id: int) -> bool:
    """
    Delete a project from the database and remove its intermediate cache directories.
    """
    project = await get_project(session, project_id)
    if project is None:
        return False

    episode_ids = [ep.id for ep in project.episodes] if project.episodes else []
    deleted = await delete_project(session, project_id)

    if deleted:
        for ep_id in episode_ids:
            ep_cache = settings.cache_dir / f"episode_{ep_id}"
            if ep_cache.exists() and ep_cache.is_dir():
                try:
                    shutil.rmtree(ep_cache, ignore_errors=True)
                    logger.debug("Deleted cache directory: {}", ep_cache)
                except Exception as exc:
                    logger.warning("Failed to clean up cache directory {}: {}", ep_cache, exc)

    return deleted
