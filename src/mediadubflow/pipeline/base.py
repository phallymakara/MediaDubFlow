"""
Pipeline stage interface.

Every processing stage must implement ``PipelineStage``.  The job manager
calls stages in sequence and persists checkpoint data after each success.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageContext:
    """
    Shared context passed between pipeline stages for a single episode.

    Stages read from and write to this object so later stages can find
    artifacts produced by earlier stages without querying the database.
    """

    episode_id: int
    project_id: int
    source_file: Path
    work_dir: Path  # Scratch directory for this episode
    output_dir: Path  # Final output directory
    source_language: str = ""
    language_confidence: float = 0.0
    extracted_audio: Path | None = None
    transcript_path: Path | None = None
    diarization_path: Path | None = None
    translation_path: Path | None = None
    subtitle_srt_path: Path | None = None
    subtitle_ass_path: Path | None = None
    tts_audio_path: Path | None = None
    mixed_audio_path: Path | None = None
    output_video_path: Path | None = None
    # Arbitrary stage-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Return value from a pipeline stage."""

    success: bool
    message: str = ""
    # Updated context fields to persist as checkpoint data
    context_updates: dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC):
    """Abstract base class for all processing stages."""

    #: Human-readable name shown in the GUI progress view.
    name: str = "Unknown Stage"

    @abstractmethod
    async def run(self, ctx: StageContext) -> StageResult:
        """
        Execute this stage.

        Args:
            ctx: Mutable episode context carrying paths and metadata.

        Returns:
            StageResult indicating success/failure and context updates.
        """
        ...

    def can_skip(self, ctx: StageContext) -> bool:
        """
        Return True if this stage's output already exists (checkpoint).

        Override in subclasses to inspect ``ctx`` for prior output paths.
        By default stages always run.
        """
        return False
