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


async def concatenate_videos(
    video_paths: list[Path],
    output_path: Path,
    *,
    timeout: float = 1800.0,
) -> tuple[int, str, str]:
    """
    Concatenate multiple video files into a single output video (.mp4).

    First attempts fast stream-copy via FFmpeg concat demuxer (-c copy).
    If stream copy fails (e.g. slight audio sample rate or codec variance),
    automatically falls back to re-encoding concat (-c:v libx264 -c:a aac).

    Args:
        video_paths: Ordered list of input video paths.
        output_path: Target merged output video path (.mp4).
        timeout: Maximum seconds to allow FFmpeg to execute (default 1800s).

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    if not video_paths:
        return 1, "", "No video files provided for concatenation"

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(video_paths) == 1:
        single = Path(video_paths[0]).resolve()
        if not single.exists() or single.stat().st_size == 0:
            return 1, "", f"Input video does not exist: {single}"
        try:
            shutil.copy2(single, output_path)
            return 0, f"Copied single video to {output_path}", ""
        except Exception as exc:
            return 1, "", f"Failed to copy video: {exc}"

    import uuid

    temp_manifest = output_path.parent / f".concat_manifest_{uuid.uuid4().hex[:8]}.txt"
    temp_output = output_path.parent / f".temp_concat_out_{uuid.uuid4().hex[:8]}.mp4"

    try:
        # Write concat demuxer manifest file
        lines = []
        for vp in video_paths:
            resolved_p = Path(vp).resolve()
            safe_str = resolved_p.as_posix().replace("'", "'\\''")
            lines.append(f"file '{safe_str}'\n")
        temp_manifest.write_text("".join(lines), encoding="utf-8")

        # 1. Attempt fast stream copy
        args_copy = [
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(temp_manifest),
            "-c",
            "copy",
            str(temp_output),
        ]
        code, stdout, stderr = await run_ffmpeg(args_copy, timeout=timeout)
        if code == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
            shutil.move(str(temp_output), str(output_path))
            return 0, stdout, stderr

        # 2. Fallback to re-encoding concat demuxer if stream copy fails
        logger.warning(
            "Fast stream-copy concatenation failed (code={}); falling back to re-encoding concat...",
            code,
        )
        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass

        args_reencode = [
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(temp_manifest),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(temp_output),
        ]
        code, stdout, stderr = await run_ffmpeg(args_reencode, timeout=timeout)
        if code == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
            shutil.move(str(temp_output), str(output_path))
            return 0, stdout, stderr

        # 3. Final fallback: filter_complex concat
        logger.warning(
            "Re-encoding concat demuxer failed (code={}); falling back to filter_complex concat...",
            code,
        )
        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass

        filter_args = ["-y"]
        for vp in video_paths:
            filter_args.extend(["-i", str(Path(vp).resolve())])
        filter_spec = (
            "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(video_paths)))
            + f"concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
        )
        filter_args.extend([
            "-filter_complex",
            filter_spec,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(temp_output),
        ])
        code, stdout, stderr = await run_ffmpeg(filter_args, timeout=timeout)
        if code == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
            shutil.move(str(temp_output), str(output_path))
            return 0, stdout, stderr

        return code, stdout, stderr
    finally:
        if temp_manifest.exists():
            try:
                temp_manifest.unlink()
            except OSError:
                pass
        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass

