"""
FFmpeg command execution utilities.

Provides non-blocking asynchronous helpers to execute FFmpeg CLI commands,
parse process results, and extract standardized audio streams.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from loguru import logger


def get_ffmpeg_path() -> str:
    """
    Locate the FFmpeg binary on the system PATH.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        FileNotFoundError: If ffmpeg is not installed or not found on PATH.
    """
    path = shutil.which("ffmpeg")
    if not path:
        raise FileNotFoundError(
            "ffmpeg executable was not found on system PATH. "
            "Please install FFmpeg and ensure it is available in your PATH."
        )
    return path


_ACTIVE_SUBPROCESSES: set[asyncio.subprocess.Process] = set()


def kill_all_ffmpeg_processes() -> None:
    """Terminate all active FFmpeg subprocesses."""
    for proc in list(_ACTIVE_SUBPROCESSES):
        try:
            proc.kill()
        except Exception:
            pass
    _ACTIVE_SUBPROCESSES.clear()


async def run_ffmpeg(
    args: list[str],
    *,
    timeout: float = 600.0,
) -> tuple[int, str, str]:
    """
    Execute an FFmpeg command asynchronously with an execution timeout.

    Args:
        args: List of command arguments passed to ffmpeg (excluding the 'ffmpeg' binary name).
        timeout: Maximum seconds to wait before terminating the subprocess (default 600.0s).

    Returns:
        Tuple of (returncode, stdout_text, stderr_text).

    Raises:
        FileNotFoundError: If ffmpeg binary is not found.
        TimeoutError: If FFmpeg execution exceeds the specified timeout.
    """
    ffmpeg_bin = get_ffmpeg_path()
    cmd = [ffmpeg_bin, *args]

    logger.debug("Executing FFmpeg command: {}", " ".join(cmd))

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _ACTIVE_SUBPROCESSES.add(process)

    try:
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            logger.error(
                "FFmpeg execution timed out after {} seconds. Terminating process...",
                timeout,
            )
            try:
                process.kill()
                await process.wait()
            except Exception as kill_exc:
                logger.warning("Failed to terminate timed out FFmpeg process: {}", kill_exc)
            raise TimeoutError(f"FFmpeg command timed out after {timeout} seconds")
        except (asyncio.CancelledError, GeneratorExit):
            logger.warning("FFmpeg execution cancelled. Killing subprocess...")
            try:
                process.kill()
                await process.wait()
            except Exception as kill_exc:
                logger.warning("Failed to kill cancelled FFmpeg process: {}", kill_exc)
            raise
    finally:
        _ACTIVE_SUBPROCESSES.discard(process)

    stdout_text = stdout_bytes.decode(errors="replace")
    stderr_text = stderr_bytes.decode(errors="replace")
    returncode = process.returncode if process.returncode is not None else 1

    if returncode != 0:
        logger.warning(
            "FFmpeg exited with code {}. Stderr snippet: {}",
            returncode,
            stderr_text[-500:] if stderr_text else "No stderr output",
        )
    else:
        logger.debug("FFmpeg completed successfully.")

    return returncode, stdout_text, stderr_text


async def extract_audio_track(
    source_file: Path,
    output_wav: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    timeout: float = 600.0,
) -> tuple[int, str, str]:
    """
    Extract a standardized PCM WAV audio track from a video file.

    Args:
        source_file: Path to the input video file.
        output_wav: Destination path for the extracted .wav file.
        sample_rate: Audio sampling frequency in Hz (default 16000 for speech models).
        channels: Number of audio channels (default 1 for mono).
        timeout: Maximum seconds to allow FFmpeg to run (default 600.0s).

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-y",  # Overwrite output without prompting
        "-i",
        str(source_file),
        "-vn",  # Disable video output
        "-acodec",
        "pcm_s16le",  # Standard 16-bit uncompressed PCM
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_wav),
    ]

    return await run_ffmpeg(args, timeout=timeout)
