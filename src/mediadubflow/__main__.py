"""Entry point for the MediaDubFlow desktop application."""

import asyncio
import sys

from mediadubflow.utils.logging import setup_logging


def main() -> None:
    """Launch the MediaDubFlow desktop application."""
    setup_logging()

    # Apply any pending database migrations before the GUI starts.
    from mediadubflow.database.session import run_db_migrations  # noqa: PLC0415

    asyncio.run(run_db_migrations())

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

    # Import Qt application only after migrations succeed so any startup
    # errors are captured in the log before the GUI appears.
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from mediadubflow.gui.app import MediaDubFlowApp  # noqa: PLC0415

    app = QApplication(sys.argv)
    app.setApplicationName("MediaDubFlow")
    app.setOrganizationName("Proseth Solutions")

    window = MediaDubFlowApp(validation_report=validation_report)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
