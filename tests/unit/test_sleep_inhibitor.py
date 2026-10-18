"""Unit tests for the SleepInhibitor utility."""

from unittest.mock import MagicMock, patch

from mediadubflow.config.settings import Settings
from mediadubflow.utils.sleep_inhibitor import SleepInhibitor


def test_sleep_inhibitor_acquire_and_release_darwin():
    """Test sleep inhibitor acquire and release on macOS."""
    inhibitor = SleepInhibitor()

    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.poll.return_value = None

    with (
        patch("sys.platform", "darwin"),
        patch("shutil.which", return_value="/usr/bin/caffeinate"),
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        # Acquire
        success = inhibitor.acquire("Unit testing")
        assert success is True
        assert inhibitor.is_active is True
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "/usr/bin/caffeinate" in args[0]
        assert "-ims" in args

        # Re-acquiring while active is a no-op
        assert inhibitor.acquire("Second call") is True

        # Release
        inhibitor.release()
        assert inhibitor.is_active is False
        mock_proc.terminate.assert_called_once()


def test_sleep_inhibitor_acquire_windows():
    """Test sleep inhibitor acquire and release on Windows."""
    inhibitor = SleepInhibitor()

    mock_ctypes = MagicMock()
    mock_ctypes.windll.kernel32.SetThreadExecutionState.return_value = 1

    with (
        patch("sys.platform", "win32"),
        patch.dict("sys.modules", {"ctypes": mock_ctypes}),
    ):
        success = inhibitor.acquire("Unit testing Windows")
        assert success is True
        assert inhibitor.is_active is True

        inhibitor.release()
        assert inhibitor.is_active is False
        assert mock_ctypes.windll.kernel32.SetThreadExecutionState.call_count == 2


def test_settings_prevent_sleep_default():
    """Verify that prevent_sleep defaults to True in Settings."""
    settings = Settings()
    assert settings.prevent_sleep is True
