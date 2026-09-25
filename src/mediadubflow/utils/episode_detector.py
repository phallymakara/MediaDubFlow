"""
Episode file detection utilities.

Scans a folder, identifies video files, extracts episode numbers from
filenames, and returns an ordered list ready for the processing queue.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

# Common video container extensions supported by FFmpeg.
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m2ts", ".webm"}
)

# Regex patterns tried in order to extract episode number from filename.
_EPISODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[Ee](?P<num>\d{1,4})"),  # E01, e12
    re.compile(r"[Ee][Pp](?P<num>\d{1,4})"),  # EP01, ep12
    re.compile(r"[Ss]\d+[Ee](?P<num>\d{1,4})"),  # S01E01
    re.compile(r"[-_\s](?P<num>\d{1,4})[-_\s.]"),  # -01-, _12_
    re.compile(r"(?P<num>\d{1,4})$"),  # trailing digits before extension
]


def detect_episodes(folder: Path) -> list[tuple[int, Path]]:
    """
    Scan *folder* for video files and return them sorted by episode number.

    Args:
        folder: Directory to scan (non-recursive).

    Returns:
        List of ``(episode_number, path)`` tuples sorted ascending.
        Files whose episode number cannot be determined are assigned
        numbers starting from 1 in alphabetical order.

    Raises:
        FileNotFoundError: If *folder* does not exist.
        NotADirectoryError: If *folder* is not a directory.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Episode folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    video_files = sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda f: f.name,
    )

    if not video_files:
        logger.warning("No video files found in {}", folder)
        return []

    results: list[tuple[int, Path]] = []
    unresolved: list[Path] = []

    for file in video_files:
        ep_num = _extract_episode_number(file.stem)
        if ep_num is not None:
            results.append((ep_num, file))
            logger.debug("Detected episode {} → {}", ep_num, file.name)
        else:
            unresolved.append(file)
            logger.warning("Could not detect episode number from filename: {}", file.name)

    # Assign sequential numbers to unresolved files based on sort order.
    existing_nums = {n for n, _ in results}
    next_num = 1
    for file in unresolved:
        while next_num in existing_nums:
            next_num += 1
        results.append((next_num, file))
        existing_nums.add(next_num)
        logger.warning("Assigned episode number {} to {}", next_num, file.name)

    return sorted(results, key=lambda t: t[0])


def _extract_episode_number(stem: str) -> int | None:
    """Try each pattern in order and return the first match."""
    for pattern in _EPISODE_PATTERNS:
        m = pattern.search(stem)
        if m:
            return int(m.group("num"))
    return None
