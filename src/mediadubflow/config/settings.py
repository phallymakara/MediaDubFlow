"""
Configuration settings loaded from environment variables and .env file.

Uses pydantic-settings to validate all configuration at startup so
misconfigurations fail fast rather than mid-pipeline.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TranslationProvider(StrEnum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class STTBackend(StrEnum):
    FASTER_WHISPER = "faster_whisper"


class TTSBackend(StrEnum):
    EDGE_TTS = "edge_tts"
    OPENAI = "openai"
    COQUI = "coqui"


class PipelineOutputMode(StrEnum):
    VOICE_DUBBING_ONLY = "voice_dubbing_only"
    SUBTITLES_ONLY = "subtitles_only"
    BOTH = "both"


class Settings(BaseSettings):
    """
    Application-wide settings resolved from environment variables and .env.

    All API keys and sensitive values must be provided via environment
    variables or a .env file — never hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./mediadubflow.db",
        description="SQLAlchemy database URL",
    )

    # --- Translation ---
    translation_provider: TranslationProvider = Field(
        default=TranslationProvider.OPENAI,
        description="LLM provider used for Khmer translation",
    )
    # OpenAI / Compatible
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_base_url: str = Field(default="", description="OpenAI custom endpoint / base URL (optional)")

    # Azure OpenAI
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint URL, e.g. https://<resource>.openai.azure.com/",
    )
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_deployment_name: str = Field(
        default="gpt-4o",
        description="Azure OpenAI deployment / model name",
    )
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API version",
    )

    # Google Gemini
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-3.8-flash", description="Google Gemini model name")

    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model name",
    )
    anthropic_base_url: str = Field(
        default="",
        description="Anthropic custom endpoint / base URL (optional)",
    )

    # --- Speech-to-Text ---
    stt_backend: STTBackend = Field(default=STTBackend.FASTER_WHISPER)
    whisper_model_size: str = Field(
        default="large-v3",
        description="Whisper model size (tiny/base/small/medium/large-v3)",
    )

    # --- Speaker Diarization ---
    hf_token: str = Field(
        default="",
        description="HuggingFace token required for pyannote.audio models",
    )

    # --- TTS ---
    tts_backend: TTSBackend = Field(default=TTSBackend.EDGE_TTS)
    tts_voice: str = Field(
        default="km-KH-PisethNeural",
        description="Khmer TTS voice name (e.g. km-KH-PisethNeural or km-KH-SreymomNeural)",
    )
    tts_model: str = Field(
        default="tts_models/km/fairseq/vits",
        description="Coqui TTS model identifier for Khmer",
    )

    # --- Output Mode ---
    output_mode: PipelineOutputMode = Field(
        default=PipelineOutputMode.VOICE_DUBBING_ONLY,
        description="Workflow mode: voice_dubbing_only | subtitles_only | both",
    )

    # --- Processing ---
    max_concurrent_episodes: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Maximum number of episodes processed in parallel",
    )
    use_gpu: bool = Field(
        default=True,
        description="Enable GPU (CUDA) acceleration when available",
    )
    device: str = Field(
        default="auto",
        description="Processing device: auto | cpu | cuda | mps",
    )

    # --- Paths ---
    output_root: Path = Field(
        default=Path("./output"),
        description="Root folder where localized episodes are saved",
    )
    cache_dir: Path = Field(
        default=Path("./.cache"),
        description="Directory for caching intermediate pipeline results",
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Directory for application and per-episode log files",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")
    log_json: bool = Field(
        default=False,
        description="Emit logs in JSON format for structured log ingestion",
    )

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Resolve 'auto' to the best available device at startup."""
        if v != "auto":
            return v
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def storage_root(self) -> Path:
        """Alias for output_root for backwards compatibility."""
        return self.output_root


# Singleton settings instance — import this from anywhere in the app.
settings = Settings()


@dataclass
class SettingsValidationReport:
    """Holds errors and warnings from startup configuration validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if there are no fatal configuration errors."""
        return len(self.errors) == 0


def ensure_directories(cfg: Settings | None = None) -> list[str]:
    """
    Ensure required application directories exist on disk.

    Creates output_root, cache_dir, and log_dir if they do not exist.
    Returns a list of error messages if any directory could not be created.
    """
    target_settings = cfg or settings
    errors: list[str] = []

    dirs_to_create = [
        ("Output directory", target_settings.output_root),
        ("Cache directory", target_settings.cache_dir),
        ("Log directory", target_settings.log_dir),
    ]

    for label, dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Failed to create {label} '{dir_path}': {exc}")

    return errors


def check_external_tools() -> list[str]:
    """
    Check for availability of external CLI tools required by the pipeline.

    Returns a list of warning messages for any missing tools.
    """
    warnings: list[str] = []
    if shutil.which("ffmpeg") is None:
        warnings.append(
            "ffmpeg was not found on PATH. Media extraction, mixing, and video rendering require ffmpeg."
        )
    if shutil.which("ffprobe") is None:
        warnings.append(
            "ffprobe was not found on PATH. Media metadata inspection requires ffprobe."
        )
    return warnings


def validate_settings(cfg: Settings | None = None) -> SettingsValidationReport:
    """
    Validate application settings and environment health before running jobs.

    Checks:
    - Required directories exist or are created.
    - Active translation provider has a non-empty API key.
    - HuggingFace token availability for speaker diarization.
    - Availability of required external tools (ffmpeg, ffprobe).
    - Database URL format.
    - Hardware acceleration settings vs actual hardware availability.
    """
    target_settings = cfg or settings
    report = SettingsValidationReport()

    # 1. Directory creation
    dir_errors = ensure_directories(target_settings)
    report.errors.extend(dir_errors)

    # 2. Database URL validation
    if not target_settings.database_url or not target_settings.database_url.strip():
        report.errors.append("Database URL cannot be empty.")
    elif not target_settings.database_url.startswith("sqlite+aiosqlite://"):
        report.warnings.append(
            f"Non-standard database URL scheme: '{target_settings.database_url}'. "
            "Ensure the appropriate async driver is installed."
        )

    # 3. Translation provider API key & endpoint validation
    if target_settings.translation_provider == TranslationProvider.OPENAI:
        if not target_settings.openai_api_key.strip():
            report.errors.append(
                "OpenAI translation provider is selected, but OPENAI_API_KEY is not configured in .env."
            )
    elif target_settings.translation_provider == TranslationProvider.AZURE_OPENAI:
        if not target_settings.azure_openai_api_key.strip():
            report.errors.append(
                "Azure OpenAI translation provider is selected, but AZURE_OPENAI_API_KEY is not configured in .env."
            )
        if not target_settings.azure_openai_endpoint.strip():
            report.errors.append(
                "Azure OpenAI translation provider is selected, but AZURE_OPENAI_ENDPOINT is not configured in .env."
            )
        if not target_settings.azure_openai_deployment_name.strip():
            report.errors.append(
                "Azure OpenAI translation provider is selected, but AZURE_OPENAI_DEPLOYMENT_NAME is not configured in .env."
            )
    elif target_settings.translation_provider == TranslationProvider.GEMINI:
        if not target_settings.gemini_api_key.strip():
            report.errors.append(
                "Gemini translation provider is selected, but GEMINI_API_KEY is not configured in .env."
            )
    elif target_settings.translation_provider == TranslationProvider.ANTHROPIC:
        if not target_settings.anthropic_api_key.strip():
            report.errors.append(
                "Anthropic translation provider is selected, but ANTHROPIC_API_KEY is not configured in .env."
            )

    # 4. Speaker diarization token warning
    if not target_settings.hf_token.strip():
        report.warnings.append(
            "HF_TOKEN is not configured. HuggingFace authentication token is required for pyannote.audio speaker diarization."
        )

    # 5. External CLI tools
    report.warnings.extend(check_external_tools())

    # 6. Hardware acceleration check
    if target_settings.device == "cuda" or (
        target_settings.use_gpu and target_settings.device == "auto"
    ):
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                report.warnings.append(
                    "CUDA GPU acceleration was requested or preferred, but torch.cuda.is_available() is False. Processing will fall back to CPU."
                )
        except ImportError:
            report.warnings.append("PyTorch is not installed; hardware acceleration check skipped.")

    return report
