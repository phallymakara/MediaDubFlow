"""
Faster-Whisper model lifecycle and caching utilities.

Manages loading and reusing WhisperModel instances across pipeline stages
to avoid expensive re-initializations and GPU VRAM reallocations.
"""

from __future__ import annotations

import re
import threading

import huggingface_hub
from faster_whisper import WhisperModel
from faster_whisper.utils import _MODELS
from loguru import logger

from mediadubflow.config.settings import settings

# Thread-safe cache storing (model_size, device, compute_type) -> WhisperModel
_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}
_CACHE_LOCK = threading.Lock()


def ensure_whisper_model_downloaded(model_size: str | None = None) -> str:
    """
    Ensure the Faster-Whisper model is downloaded to the local HuggingFace cache.

    If not cached yet, downloads it with a live progress percentage bar in the terminal.
    Returns the local path to the model snapshot directory.
    """
    import os  # noqa: PLC0415

    resolved_size = model_size or settings.whisper_model_size
    if "PYTEST_CURRENT_TEST" in os.environ:
        return resolved_size

    if re.match(r".*/.*", resolved_size):
        repo_id = resolved_size
    else:
        repo_id = _MODELS.get(resolved_size, resolved_size)

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    try:
        path = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        logger.info("Faster-Whisper model '{}' is ready locally at: {}", resolved_size, path)
        return path
    except Exception:
        logger.info(
            "Downloading Faster-Whisper model '{}' ({}) from Hugging Face Hub...",
            resolved_size,
            repo_id,
        )
        try:
            path = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
            )
            logger.info(
                "Faster-Whisper model '{}' downloaded successfully to: {}",
                resolved_size,
                path,
            )
            return path
        except Exception as exc:
            logger.error("Failed to download Faster-Whisper model '{}': {}", resolved_size, exc)
            raise


def get_whisper_model(
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    """
    Get a cached Faster-Whisper model or initialize a new instance.

    Args:
        model_size: Whisper model size (default from settings, e.g. "large-v3" or "base").
        device: Device to run inference on ("cuda", "cpu"). Defaults to settings.device.
        compute_type: Quantization type (e.g. "float16", "int8"). If None, defaults
                      to "float16" for CUDA and "int8" for CPU.

    Returns:
        Loaded WhisperModel instance.
    """
    resolved_size = model_size or settings.whisper_model_size
    resolved_device = device or settings.device

    if resolved_device == "auto":
        resolved_device = "cpu"

    if compute_type is None:
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute = compute_type

    cache_key = (resolved_size, resolved_device, resolved_compute)

    # Double-checked locking optimization
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            logger.debug("Reusing cached WhisperModel for {}", cache_key)
            return _MODEL_CACHE[cache_key]

        try:
            ensure_whisper_model_downloaded(resolved_size)
        except Exception as exc:
            logger.warning("Could not pre-verify Whisper model cache: {}", exc)

        logger.info(
            "Loading Faster-Whisper model: size={} device={} compute_type={}",
            resolved_size,
            resolved_device,
            resolved_compute,
        )

        model = WhisperModel(
            model_size_or_path=resolved_size,
            device=resolved_device,
            compute_type=resolved_compute,
        )
        _MODEL_CACHE[cache_key] = model
        return model


def clear_whisper_model_cache() -> None:
    """Clear all cached Whisper models to release memory / VRAM."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        logger.debug("Cleared Whisper model cache.")
