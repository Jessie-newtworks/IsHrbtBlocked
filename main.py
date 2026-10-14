"""IsHrbtBlocked - entry point.

    python main.py               normal start (click "Start Logging")
    python main.py --autostart   start logging automatically (used by the
                                 Debian boot launcher, run_hrbt.sh)
"""

import argparse
import sys
import threading

from hrbt import app_dir
from hrbt.error_log import log_error

LOCK_NAME = ".ishrbtblocked.lock"


def _excepthook(exc_type, exc, tb):
    log_error("Unhandled exception.", exc)
    sys.__excepthook__(exc_type, exc, tb)


def _thread_excepthook(args):
    log_error(f"Unhandled exception in thread '{args.thread.name if args.thread else '?'}'.",
              args.exc_value)


def acquire_single_instance_lock():
    """Make sure only one copy runs, so two copies never write the same Excel
    file. Returns the open lock file (keep it open) or None if another copy
    already holds the lock. The OS releases the lock if the app crashes."""
    path = app_dir() / LOCK_NAME
    fh = open(path, "a+")
    try:
        if sys.platform.startswith("win"):
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def main() -> int:
    parser = argparse.ArgumentParser(description="HRBT tunnel-exit stoppage monitor")
    parser.add_argument("--autostart", action="store_true",
                        help="start logging automatically once the window opens")
    args = parser.parse_args()

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    lock = acquire_single_instance_lock()
    if lock is None:
        msg = "IsHrbtBlocked is already running."
        log_error(msg + " A second copy was not started.", level="INFO")
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("IsHrbtBlocked", msg)
            root.destroy()
        except Exception:
            print(msg)
        return 0

    from hrbt.gui import App
    App(autostart=args.autostart).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
