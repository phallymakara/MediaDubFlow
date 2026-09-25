"""Package init — re-exports the singleton settings instance."""

from mediadubflow.config.settings import (
    Settings,
    SettingsValidationReport,
    ensure_directories,
    settings,
    validate_settings,
)

__all__ = [
    "Settings",
    "SettingsValidationReport",
    "ensure_directories",
    "settings",
    "validate_settings",
]
