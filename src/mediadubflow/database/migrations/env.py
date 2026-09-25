"""
Alembic migration environment — async SQLAlchemy configuration.

This file is invoked by Alembic for every migration command.
It connects to the database using the same async engine and URL
as the application, loaded from settings (which reads from .env).

References:
    https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from mediadubflow.config import settings
from mediadubflow.models.orm import Base

# Alembic Config object — provides access to alembic.ini values.
config = context.config

# Configure Python logging from alembic.ini [loggers] section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Provide our full metadata to autogenerate so Alembic can diff the schema.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (no live DB connection needed).

    Generates SQL script output instead of executing against a database.
    Useful for producing migration scripts to review before applying.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render enum types as their string values for SQLite compatibility.
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):  # type: ignore[no-untyped-def]
    """Configure context and run migrations on a live sync connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # render_as_batch is required for SQLite ALTER TABLE support.
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode using the async engine.

    Creates a fresh engine from settings.database_url (loaded from .env)
    so that the migration uses the same database as the application.
    The connection is run synchronously inside run_sync() because Alembic's
    internal migration runner is not async-aware.
    """
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — runs the async function."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
