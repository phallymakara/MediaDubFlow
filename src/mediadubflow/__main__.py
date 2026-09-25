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
