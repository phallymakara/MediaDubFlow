"""Unit tests for configuration loading and startup settings validation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from mediadubflow.config.settings import (
    Settings,
    SettingsValidationReport,
    TranslationProvider,
    check_external_tools,
    ensure_directories,
    validate_settings,
)


def test_settings_default_values() -> None:
    """Settings should have reasonable default values."""
    cfg = Settings(_env_file=None)
    assert cfg.database_url.startswith("sqlite+aiosqlite://")
    assert cfg.translation_provider == TranslationProvider.OPENAI
    assert cfg.max_concurrent_episodes == 2
    assert cfg.log_level == "INFO"
    assert cfg.log_json is False


def test_settings_env_override(monkeypatch) -> None:
    """Settings should be overridable via environment variables."""
    monkeypatch.setenv("MAX_CONCURRENT_EPISODES", "4")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "anthropic")

    cfg = Settings(_env_file=None)
    assert cfg.max_concurrent_episodes == 4
    assert cfg.log_level == "DEBUG"
    assert cfg.translation_provider == TranslationProvider.ANTHROPIC


def test_ensure_directories_creates_folders(tmp_path: Path) -> None:
    """ensure_directories creates output_root, cache_dir, and log_dir if they do not exist."""
    out_dir = tmp_path / "custom_output"
    cache_dir = tmp_path / "custom_cache"
    log_dir = tmp_path / "custom_logs"

    cfg = Settings(
        _env_file=None,
        output_root=out_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
    )

    errors = ensure_directories(cfg)
    assert errors == []
    assert out_dir.is_dir()
    assert cache_dir.is_dir()
    assert log_dir.is_dir()


def test_ensure_directories_handles_os_error(tmp_path: Path) -> None:
    """ensure_directories returns error strings if folder creation fails."""
    cfg = Settings(_env_file=None, output_root=tmp_path / "out")

    with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
        errors = ensure_directories(cfg)
        assert len(errors) == 3
        assert any("Permission denied" in err for err in errors)


def test_check_external_tools_present() -> None:
    """check_external_tools returns no warnings when ffmpeg and ffprobe are on PATH."""
    with patch("shutil.which", return_value="/usr/bin/mock-tool"):
        warnings = check_external_tools()
        assert warnings == []


def test_check_external_tools_missing() -> None:
    """check_external_tools returns warnings when ffmpeg and ffprobe are missing."""
    with patch("shutil.which", return_value=None):
        warnings = check_external_tools()
        assert len(warnings) == 2
        assert any("ffmpeg was not found" in w for w in warnings)
        assert any("ffprobe was not found" in w for w in warnings)


def test_validate_settings_valid_openai(tmp_path: Path) -> None:
    """validate_settings produces a valid report when OpenAI API key is supplied."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="sk-test-key",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report: SettingsValidationReport = validate_settings(cfg)
        assert report.is_valid
        assert len(report.errors) == 0
        assert len(report.warnings) == 0


def test_validate_settings_missing_openai_key(tmp_path: Path) -> None:
    """validate_settings flags an error when OpenAI is selected but key is missing."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert not report.is_valid
        assert any("OPENAI_API_KEY is not configured" in err for err in report.errors)


def test_validate_settings_missing_anthropic_key(tmp_path: Path) -> None:
    """validate_settings flags an error when Anthropic is selected but key is missing."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.ANTHROPIC,
        anthropic_api_key="",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert not report.is_valid
        assert any("ANTHROPIC_API_KEY is not configured" in err for err in report.errors)


def test_validate_settings_valid_anthropic(tmp_path: Path) -> None:
    """validate_settings passes when Anthropic is selected and key is supplied."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.ANTHROPIC,
        anthropic_api_key="sk-ant-test-key",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert report.is_valid
        assert len(report.errors) == 0


def test_validate_settings_missing_gemini_key(tmp_path: Path) -> None:
    """validate_settings fails when Gemini is selected and GEMINI_API_KEY is missing."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.GEMINI,
        gemini_api_key="",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert not report.is_valid
        assert any("GEMINI_API_KEY is not configured" in err for err in report.errors)


def test_validate_settings_valid_gemini(tmp_path: Path) -> None:
    """validate_settings passes when Gemini is selected and key is supplied."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.GEMINI,
        gemini_api_key="AIzaSy-test-key",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert report.is_valid
        assert len(report.errors) == 0


def test_validate_settings_missing_azure_openai_fields(tmp_path: Path) -> None:
    """validate_settings fails when Azure OpenAI is selected but required fields are missing."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.AZURE_OPENAI,
        azure_openai_api_key="",
        azure_openai_endpoint="",
        azure_openai_deployment_name="",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert not report.is_valid
        assert any("AZURE_OPENAI_API_KEY is not configured" in err for err in report.errors)
        assert any("AZURE_OPENAI_ENDPOINT is not configured" in err for err in report.errors)
        assert any("AZURE_OPENAI_DEPLOYMENT_NAME is not configured" in err for err in report.errors)


def test_validate_settings_valid_azure_openai(tmp_path: Path) -> None:
    """validate_settings passes when Azure OpenAI is selected with all required fields."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.AZURE_OPENAI,
        azure_openai_api_key="azure-test-key",
        azure_openai_endpoint="https://myresource.openai.azure.com/",
        azure_openai_deployment_name="gpt-4o",
        hf_token="hf_test_token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert report.is_valid
        assert len(report.errors) == 0


def test_validate_settings_missing_hf_token_is_warning(tmp_path: Path) -> None:
    """Missing HF_TOKEN produces a warning but does not invalidate startup configuration."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="sk-test-key",
        hf_token="",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert report.is_valid  # Still valid because hf_token is a warning
        assert any("HF_TOKEN is not configured" in w for w in report.warnings)


def test_validate_settings_empty_database_url(tmp_path: Path) -> None:
    """Empty database URL produces a validation error."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        database_url="",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="sk-test-key",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert not report.is_valid
        assert any("Database URL cannot be empty" in err for err in report.errors)


def test_validate_settings_non_standard_database_url(tmp_path: Path) -> None:
    """Non-standard database URL produces a warning."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="sk-test-key",
        hf_token="token",
        device="cpu",
    )

    with patch("shutil.which", return_value="/usr/bin/tool"):
        report = validate_settings(cfg)
        assert report.is_valid
        assert any("Non-standard database URL scheme" in w for w in report.warnings)


def test_validate_settings_cuda_fallback_warning(tmp_path: Path) -> None:
    """Requesting CUDA when CUDA is unavailable logs a warning."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="sk-test-key",
        hf_token="token",
        device="cuda",
    )

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with (
        patch("shutil.which", return_value="/usr/bin/tool"),
        patch.dict(sys.modules, {"torch": mock_torch}),
    ):
        report = validate_settings(cfg)
        assert report.is_valid
        assert any("torch.cuda.is_available() is False" in w for w in report.warnings)


def test_validate_settings_pytorch_missing_warning(tmp_path: Path) -> None:
    """When PyTorch is not installed, a hardware acceleration notice is logged."""
    cfg = Settings(
        _env_file=None,
        output_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        translation_provider=TranslationProvider.OPENAI,
        openai_api_key="sk-test-key",
        hf_token="token",
        device="cuda",
    )

    with (
        patch("shutil.which", return_value="/usr/bin/tool"),
        patch.dict(sys.modules, {"torch": None}),
    ):
        report = validate_settings(cfg)
        assert report.is_valid
        assert any("PyTorch is not installed" in w for w in report.warnings)
