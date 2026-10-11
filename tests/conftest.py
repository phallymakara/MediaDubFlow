"""Global pytest fixtures and test setup for MediaDubFlow."""

from __future__ import annotations

from typing import Generator

import pytest

from mediadubflow.config.settings import PipelineOutputMode, settings


@pytest.fixture(autouse=True)
def reset_global_settings() -> Generator[None, None, None]:
    """Reset mutable global settings before and after each test to prevent test leakage."""
    original_mode = settings.output_mode
    settings.output_mode = PipelineOutputMode.BOTH
    try:
        yield
    finally:
        settings.output_mode = original_mode
