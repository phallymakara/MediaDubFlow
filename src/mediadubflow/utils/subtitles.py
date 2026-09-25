"""
Subtitle generation utilities for SRT and ASS formats.

Builds standardized SubRip (.srt) and styled Advanced SubStation Alpha (.ass)
subtitle files from timestamped transcript/translation segments.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import ass
import pysrt
from loguru import logger


def create_srt_file(
    segments: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """
    Generate a standard SubRip (.srt) subtitle file.

    Args:
        segments: List of segment dicts with 'start', 'end', and 'translated_text' or 'text'.
        output_path: Destination path for the .srt file.

    Returns:
        The output Path instance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subs = pysrt.SubRipFile()

    for idx, seg in enumerate(segments, start=1):
        text = str(
            seg.get("translated_text") or seg.get("text") or seg.get("original_text", "")
        ).strip()

        start_sec = max(0.0, float(seg.get("start", 0.0)))
        end_sec = max(start_sec + 0.1, float(seg.get("end", start_sec + 1.0)))

        item = pysrt.SubRipItem(
            index=idx,
            start=pysrt.SubRipTime(seconds=start_sec),
            end=pysrt.SubRipTime(seconds=end_sec),
            text=text,
        )
        subs.append(item)

    subs.save(str(output_path), encoding="utf-8")
    logger.debug("Generated SRT subtitles at '{}' with {} cues", output_path, len(subs))
    return output_path


def create_ass_file(
    segments: list[dict[str, Any]],
    output_path: Path,
    *,
    font_name: str = "Kantumruy Pro",
    font_size: float = 48.0,
) -> Path:
    """
    Generate an Advanced SubStation Alpha (.ass) subtitle file.

    Configured with 1080p canvas coordinates, bottom-center alignment,
    crisp outline without drop shadows, and modern Khmer typography.

    Args:
        segments: List of segment dicts with 'start', 'end', and 'translated_text' or 'text'.
        output_path: Destination path for the .ass file.
        font_name: Subtitle font name (default: "Kantumruy Pro").
        font_size: Point size of subtitles on a 1080p canvas (default: 48.0).

    Returns:
        The output Path instance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ass.Document()

    # Script Info Header
    doc.info["Title"] = output_path.stem
    doc.info["ScriptType"] = "v4.00+"
    doc.info["WrapStyle"] = "0"
    doc.info["ScaledBorderAndShadow"] = "yes"
    doc.info["PlayResX"] = "1920"
    doc.info["PlayResY"] = "1080"

    # Default Style for Khmer Subtitles
    style = ass.Style(
        name="KhmerDefault",
        fontname=font_name,
        fontsize=font_size,
        primary_color=ass.data.Color(r=255, g=255, b=255, a=0),
        secondary_color=ass.data.Color(r=255, g=255, b=255, a=0),
        outline_color=ass.data.Color(r=0, g=0, b=0, a=0),
        back_color=ass.data.Color(r=0, g=0, b=0, a=0),
        bold=True,
        italic=False,
        underline=False,
        strike_out=False,
        scale_x=100.0,
        scale_y=100.0,
        spacing=0.0,
        angle=0.0,
        border_style=1,
        outline=2.5,
        shadow=0.0,  # Strict compliance: do not use drop shadows
        alignment=2,  # Bottom Center
        margin_l=60,
        margin_r=60,
        margin_v=45,
        encoding=1,
    )
    doc.styles.append(style)

    for seg in segments:
        text = str(
            seg.get("translated_text") or seg.get("text") or seg.get("original_text", "")
        ).strip()

        start_sec = max(0.0, float(seg.get("start", 0.0)))
        end_sec = max(start_sec + 0.1, float(seg.get("end", start_sec + 1.0)))

        event = ass.Dialogue(
            style="KhmerDefault",
            start=timedelta(seconds=start_sec),
            end=timedelta(seconds=end_sec),
            text=text,
        )
        doc.events.append(event)

    with output_path.open("w", encoding="utf_8_sig") as f:
        doc.dump_file(f)

    logger.debug("Generated ASS subtitles at '{}' with {} events", output_path, len(doc.events))
    return output_path
