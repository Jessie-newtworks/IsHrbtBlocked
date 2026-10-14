"""Plain-text error log (errors.log).

Every error the app hits - stream failures, Excel write failures, bad settings,
uncaught exceptions - is appended here with a timestamp. Safe to call from any
thread.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path

from . import app_dir

ERROR_LOG_NAME = "errors.log"
EVENT_LOG_NAME = "events.log"

_lock = threading.Lock()
_log_path: Path = app_dir() / ERROR_LOG_NAME


def set_log_path(path: Path) -> None:
    """Point the error log somewhere else (used by tests). events.log goes
    in the same folder."""
    global _log_path
    _log_path = Path(path)


def get_log_path() -> Path:
    return _log_path


def log_error(message: str, exc: BaseException | None = None, level: str = "ERROR") -> None:
    """Append a timestamped error line (plus traceback, if given) to errors.log."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {level}: {message}\n"
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        text += "".join(f"    {line}\n" for line in tb.rstrip().splitlines())
    with _lock:
        try:
            with open(_log_path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            # Last resort - never let error logging crash the app.
            print(text, end="")


def log_event(message: str) -> None:
    """Append a line to events.log: one line per event start / end, so the
    daily totals in Excel can be checked against what really happened."""
    with _lock:
        try:
            with open(_log_path.parent / EVENT_LOG_NAME, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            print(message)
