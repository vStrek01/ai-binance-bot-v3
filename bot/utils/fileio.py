"""Utility helpers for safe cross-platform file I/O."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Optional

if os.name == "nt":  # pragma: no cover - Windows-only import
    import msvcrt
else:  # pragma: no cover - Unix-only import
    import fcntl  # type: ignore[import]


class FileLock:
    """Small cross-platform advisory file lock."""

    def __init__(self, path: Path, timeout: float | None = 10.0, poll_interval: float = 0.05) -> None:
        self._path = Path(path)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._handle: Optional[BinaryIO] = None

    def acquire(self) -> None:
        start = time.time()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            handle: BinaryIO
            try:
                handle = open(self._path, "a+b")
            except OSError as exc:  # File could not be opened yet
                if self._timeout is not None and (time.time() - start) >= self._timeout:
                    raise TimeoutError(f"Unable to open lock file {self._path}: {exc}") from exc
                time.sleep(self._poll_interval)
                continue
            try:
                self._lock_handle(handle)
                self._handle = handle
                return
            except OSError as exc:
                handle.close()
                if self._timeout is not None and (time.time() - start) >= self._timeout:
                    raise TimeoutError(f"Timed out acquiring lock for {self._path}: {exc}") from exc
                time.sleep(self._poll_interval)

    def release(self) -> None:
        if not self._handle:
            return
        try:
            self._unlock_handle(self._handle)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:  # pragma: no cover - trivial wrapper
        del exc_type, exc, tb
        self.release()

    if os.name == "nt":  # pragma: no cover - Windows-specific logic

        @staticmethod
        def _lock_handle(handle: BinaryIO) -> None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

        @staticmethod
        def _unlock_handle(handle: BinaryIO) -> None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    else:  # pragma: no cover - Unix-specific logic

        @staticmethod
        def _lock_handle(handle: BinaryIO) -> None:
            fcntl.flock(handle, fcntl.LOCK_EX)

        @staticmethod
        def _unlock_handle(handle: BinaryIO) -> None:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write_text(path: Path, payload: str, encoding: str = "utf-8") -> None:
    """Atomically write text, with a Windows fallback when renames are denied."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as tmp_file:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        try:
            os.replace(tmp_name, target)
            return
        except PermissionError as exc:
            if os.name == "nt":
                try:
                    _rewrite_in_place(target, payload, encoding)
                    return
                except OSError:
                    raise exc
            raise
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass


def _rewrite_in_place(target: Path, payload: str, encoding: str) -> None:
    """Best-effort fallback when Windows blocks os.replace due to file locks."""
    with open(target, "w", encoding=encoding) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
