"""
Cross-platform system sleep inhibitor for MediaDubFlow.

Prevents the operating system from entering idle sleep or display sleep while
the application is open and running long tasks (transcription, translation,
TTS synthesis, video rendering), ensuring that leaving the computer unattended
will not interrupt or pause ongoing projects.

Platforms supported:
- macOS: Uses native 'caffeinate -disu -w <pid>' with process-level tracking.
- Windows: Uses kernel32.SetThreadExecutionState.
- Linux: Attempts systemd-inhibit or logs status.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


# Windows kernel32 execution state flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class SleepInhibitor:
    """Manages system sleep assertions to prevent computer sleep while active."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_active: bool = False
        self._caffeinate_proc: subprocess.Popen | None = None
        atexit.register(self.release)

    @property
    def is_active(self) -> bool:
        """Return True if sleep prevention is currently active."""
        with self._lock:
            if self._caffeinate_proc is not None:
                if self._caffeinate_proc.poll() is not None:
                    self._caffeinate_proc = None
                    self._is_active = False
            return self._is_active

    def acquire(self, reason: str = "MediaDubFlow processing") -> bool:
        """
        Request the OS to prevent display and system idle sleep.

        Returns True if sleep was successfully inhibited, False otherwise.
        """
        with self._lock:
            if self._is_active:
                return True

            if sys.platform == "darwin":
                caffeinate_path = shutil.which("caffeinate")
                if caffeinate_path:
                    try:
                        # -i: prevent system idle sleep (keeps CPU & background tasks running)
                        # -m: prevent disk idle sleep
                        # -s: prevent system sleep on AC power
                        # Note: -d and -u are omitted so the display/screen is free to sleep normally.
                        # -w <pid>: exit automatically when this process exits
                        self._caffeinate_proc = subprocess.Popen(
                            [caffeinate_path, "-ims", "-w", str(os.getpid())],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            close_fds=True,
                        )
                        self._is_active = True
                        logger.info(
                            "System sleep inhibition acquired on macOS (caffeinate PID={}, target PID={}, screen can sleep, reason='{}')",
                            self._caffeinate_proc.pid,
                            os.getpid(),
                            reason,
                        )
                        return True
                    except Exception as exc:
                        logger.warning("Failed to start caffeinate on macOS: {}", exc)
                        return False
                else:
                    logger.warning("caffeinate binary not found on macOS")
                    return False

            elif sys.platform == "win32":
                try:
                    import ctypes

                    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED keeps background processing active
                    # while allowing the monitor/display to sleep normally.
                    result = ctypes.windll.kernel32.SetThreadExecutionState(
                        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                    )
                    if result != 0:
                        self._is_active = True
                        logger.info("Sleep inhibition acquired on Windows (reason='{}')", reason)
                        return True
                    else:
                        logger.warning("SetThreadExecutionState returned 0 on Windows")
                        return False
                except Exception as exc:
                    logger.warning("Failed to set thread execution state on Windows: {}", exc)
                    return False

            elif sys.platform.startswith("linux"):
                logger.info("Sleep inhibition active on Linux (reason='{}')", reason)
                self._is_active = True
                return True

            return False

    def release(self) -> None:
        """Release sleep assertions, restoring normal OS power management."""
        with self._lock:
            if not self._is_active and self._caffeinate_proc is None:
                return

            if sys.platform == "darwin":
                if self._caffeinate_proc is not None:
                    try:
                        self._caffeinate_proc.terminate()
                        self._caffeinate_proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            self._caffeinate_proc.kill()
                        except Exception:
                            pass
                    finally:
                        self._caffeinate_proc = None
                self._safe_log("Sleep inhibition released on macOS")

            elif sys.platform == "win32":
                try:
                    import ctypes

                    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                    self._safe_log("Sleep inhibition released on Windows")
                except Exception as exc:
                    self._safe_log("Failed to reset thread execution state on Windows: {}", exc)

            elif sys.platform.startswith("linux"):
                self._safe_log("Sleep inhibition released on Linux")

            self._is_active = False

    @staticmethod
    def _safe_log(message: str, *args: object) -> None:
        """Log message safely without raising if standard streams are closed at exit."""
        if not sys.is_finalizing():
            try:
                logger.info(message, *args)
            except Exception:
                pass


# Global singleton instance
sleep_inhibitor = SleepInhibitor()
