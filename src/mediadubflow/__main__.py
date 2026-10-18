"""Entry point for the MediaDubFlow desktop application."""

import asyncio
import sys

from mediadubflow.utils.logging import setup_logging


def main() -> None:
    """Launch the MediaDubFlow desktop application."""
    setup_logging()

    # Apply any pending database migrations before the GUI starts.
    from mediadubflow.database.repository import reconcile_orphaned_episodes  # noqa: PLC0415
    from mediadubflow.database.session import get_session, run_db_migrations  # noqa: PLC0415

    async def _init_db() -> None:
        await run_db_migrations()
        async with get_session() as session:
            await reconcile_orphaned_episodes(session)

    asyncio.run(_init_db())

    # Validate settings and system environment at startup.
    from loguru import logger  # noqa: PLC0415

    from mediadubflow.config.settings import settings, validate_settings  # noqa: PLC0415

    validation_report = validate_settings(settings)
    for warning in validation_report.warnings:
        logger.warning("Startup configuration warning: {}", warning)
    for error in validation_report.errors:
        logger.error("Startup configuration error: {}", error)

    # Ensure Faster-Whisper model is downloaded before launching GUI, showing terminal progress
    from mediadubflow.utils.whisper_model import ensure_whisper_model_downloaded  # noqa: PLC0415

    try:
        ensure_whisper_model_downloaded()
    except Exception as exc:
        logger.warning("Could not pre-download Whisper model at startup: {}", exc)

    # Ensure Qt platform plugins (qwindows.dll) can be located even in deep/OneDrive paths.
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import PySide6  # noqa: PLC0415

    plugins_dir = Path(PySide6.__file__).resolve().parent / "plugins" / "platforms"
    if plugins_dir.exists() and "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_dir)

    # Ensure sleep prevention is activated if configured so the system doesn't sleep
    # during long rendering, translation, or audio tasks while user is away.
    from mediadubflow.utils.sleep_inhibitor import sleep_inhibitor  # noqa: PLC0415

    if settings.prevent_sleep:
        sleep_inhibitor.acquire("MediaDubFlow desktop application active")

    # Import Qt application only after migrations succeed so any startup
    # errors are captured in the log before the GUI appears.
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from mediadubflow.gui.app import MediaDubFlowApp  # noqa: PLC0415

    app = QApplication(sys.argv)
    app.setApplicationName("MediaDubFlow")
    app.setOrganizationName("Proseth Solutions")
    app.aboutToQuit.connect(sleep_inhibitor.release)

    window = MediaDubFlowApp(validation_report=validation_report)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
