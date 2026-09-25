"""
Async SQLAlchemy engine, session factory, and database initialisation.

On first launch (and on every launch) call run_db_migrations() to apply
any pending Alembic migrations.  This is idempotent — if the schema is
already up to date Alembic does nothing.

Usage:
    await run_db_migrations()          # once at startup

    async with get_session() as session:
        result = await session.execute(...)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mediadubflow.config import settings

_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

_SessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def run_db_migrations() -> None:
    """
    Apply all pending Alembic migrations to the database.

    Runs 'alembic upgrade head' programmatically using the same database
    URL as the rest of the application (loaded from settings / .env).
    Safe to call on every application launch — Alembic skips migrations
    that have already been applied.

    Raises:
        Exception: Any Alembic or database error is logged then re-raised
                   so the application can fail fast at startup rather than
                   encountering a broken schema mid-pipeline.
    """
    import asyncio  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415

    # Locate alembic.ini relative to this file's package root.
    ini_path = Path(__file__).parents[4] / "alembic.ini"

    alembic_cfg = Config(str(ini_path))

    logger.info("Applying database migrations (alembic upgrade head)")
    try:
        # Alembic's command.upgrade is synchronous; run it in a thread so
        # we don't block the asyncio event loop during startup.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, command.upgrade, alembic_cfg, "head")
        logger.info("Database schema is up to date")
    except Exception as exc:
        logger.error("Database migration failed: {}", exc)
        raise


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession]:
    """
    Provide a transactional async database session.

    Commits on clean exit, rolls back on any exception.
    """
    async with _SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
