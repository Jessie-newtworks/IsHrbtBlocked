"""IsHrbtBlocked - monitors the HRBT tunnel exit cameras for traffic stoppages."""

import sys
from pathlib import Path

__version__ = "1.0.0"


def app_dir() -> Path:
    """Folder that holds config.json, the Excel log, errors.log and the model.

    When frozen into an executable (e.g. PyInstaller) this is the folder of the
    .exe; otherwise it is the project root (the folder containing main.py).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
