"""
Faster-Whisper model lifecycle and caching utilities.

Manages loading and reusing WhisperModel instances across pipeline stages
to avoid expensive re-initializations and GPU VRAM reallocations.
"""

from __future__ import annotations

import threading

from faster_whisper import WhisperModel
from loguru import logger

from mediadubflow.config.settings import settings

# Thread-safe cache storing (model_size, device, compute_type) -> WhisperModel
_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}
_CACHE_LOCK = threading.Lock()


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

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            logger.debug("Reusing cached WhisperModel for {}", cache_key)
            return _MODEL_CACHE[cache_key]

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
