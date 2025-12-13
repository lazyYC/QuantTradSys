import os
import errno
import psutil
import logging
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

class AlreadyRunningError(Exception):
    """Raised when a process is already running with the same lock."""
    pass

class PIDLock:
    """
    A simple PID-based lock file mechanism.
    Checks if the PID in the lock file is actually running via psutil.
    """
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file

    def acquire(self) -> None:
        """
        Acquire the lock.
        If lock file exists:
            - Check if PID is alive.
            - If alive: Raise AlreadyRunningError.
            - If dead: Overwrite lock file (stale lock).
        If lock file doesn't exist:
            - Create it and write current PID.
        """
        if self.lock_file.exists():
            try:
                pid = int(self.lock_file.read_text().strip())
                if psutil.pid_exists(pid):
                    raise AlreadyRunningError(f"Process {pid} is already running with lock {self.lock_file}")
                else:
                    LOGGER.warning(f"Found stale lock file {self.lock_file} for PID {pid}. Overwriting.")
            except ValueError:
                 LOGGER.warning(f"Found corrupted lock file {self.lock_file}. Overwriting.")
            except OSError as e:
                # Handle race condition if file is deleted between exists() and read()
                if e.errno != errno.ENOENT:
                     raise

        # Write current PID
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.write_text(str(os.getpid()))
        LOGGER.info(f"Acquired lock {self.lock_file} for PID {os.getpid()}")

    def release(self) -> None:
        """
        Release the lock if it belongs to this process.
        """
        if not self.lock_file.exists():
            return

        try:
            pid = int(self.lock_file.read_text().strip())
            if pid == os.getpid():
                self.lock_file.unlink()
                LOGGER.info(f"Released lock {self.lock_file}")
        except (ValueError, OSError, FileNotFoundError):
            pass
