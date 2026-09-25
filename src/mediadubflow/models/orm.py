"""
SQLAlchemy ORM models for projects, episodes, and processing jobs.

Schema evolves via Alembic migrations — do not alter tables directly.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class EpisodeStatus(enum.StrEnum):
    """Processing lifecycle states for a single episode."""

    PENDING = "pending"
    QUEUED = "queued"
    DETECTING_LANGUAGE = "detecting_language"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DETECTING_SPEAKERS = "detecting_speakers"
    TRANSLATING = "translating"
    REVIEW = "review"
    GENERATING_SUBTITLES = "generating_subtitles"
    GENERATING_TTS = "generating_tts"
    MIXING_AUDIO = "mixing_audio"
    RENDERING = "rendering"
    QC = "qc"
    DONE = "done"
    FAILED = "failed"
    PAUSED = "paused"


class Project(Base):
    """A localization project containing one or more episodes."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_folder: Mapped[str] = mapped_column(String(1024), nullable=False)
    output_folder: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    target_language: Mapped[str] = mapped_column(String(10), default="km")
    glossary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    episodes: Mapped[list[Episode]] = relationship(
        "Episode", back_populates="project", cascade="all, delete-orphan"
    )


class Episode(Base):
    """A single episode belonging to a project."""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    episode_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_file: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[EpisodeStatus] = mapped_column(
        Enum(EpisodeStatus), default=EpisodeStatus.PENDING, nullable=False
    )
    detected_language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Checkpoint paths for intermediate artifacts
    extracted_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    transcript_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    diarization_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    translation_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    subtitle_srt_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    subtitle_ass_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    tts_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    mixed_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    output_video_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    project: Mapped[Project] = relationship("Project", back_populates="episodes")
    speakers: Mapped[list[Speaker]] = relationship(
        "Speaker", back_populates="episode", cascade="all, delete-orphan"
    )


class Speaker(Base):
    """
    A detected speaker within an episode, with an assigned Khmer voice.

    Voice assignment is persisted so the same speaker always uses the same
    voice across multiple episodes of a series.
    """

    __tablename__ = "speakers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "SPEAKER_00"
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    tts_voice_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    episode: Mapped[Episode] = relationship("Episode", back_populates="speakers")
