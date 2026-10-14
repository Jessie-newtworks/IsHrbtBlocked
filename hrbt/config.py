"""Settings stored in config.json.

User-editable settings (also editable from the GUI):
    stream_1_url             - live stream URL for stream 1
    stream_2_url             - live stream URL for stream 2
    event_threshold_seconds  - seconds with no passing vehicle before an event is logged
    window_start             - "HHMM", 24-hour time when event logging begins (e.g. "0600")
    window_end               - "HHMM", 24-hour time when event logging ends   (e.g. "2000")

Advanced settings (JSON only):
    stream_1_name / stream_2_name - labels shown in the GUI
    excel_file                    - name of the Excel log, kept next to main.py
    model                         - YOLO weights file (downloaded automatically)
    confidence                    - minimum YOLO confidence for a vehicle detection
    motion_pixel_fraction         - every frame is checked. Frames that barely differ
                                    from the previous one (fewer than this fraction of
                                    pixels changed) can't contain a passing car, so YOLO
                                    is skipped for them. 0 = run YOLO on every frame
    ignore_stationary_vehicles    - true: only *moving* vehicles count as "passing",
                                    so a parked / disabled vehicle in view can't hide
                                    a stoppage (cameras watch the tunnel exits, so a
                                    stoppage in the tunnel = no cars coming out)
    stale_stream_seconds          - if no new frames arrive for this long the feed is
                                    treated as down (that time is never counted)
    show_detection_boxes          - draw YOLO boxes on the live view (green = moving,
                                    orange = stopped, grey = too far away, ignored)
    min_vehicle_size              - ignore detections smaller than this fraction of the
                                    picture height (0.06 = ~14 px at 320x240). Too
                                    small and far-away false hits count; too big and a
                                    fast car can cross the counted zone between the
                                    camera's 2-4 real pictures per second unseen
    detection_image_size          - size YOLO scales each picture to (640). 960 finds
                                    small / distant cars sooner but uses ~2x the CPU
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import app_dir

CONFIG_NAME = "config.json"

PLACEHOLDER_STREAM_1 = "PASTE_STREAM_1_URL_HERE"
PLACEHOLDER_STREAM_2 = "PASTE_STREAM_2_URL_HERE"

_HHMM = re.compile(r"^\d{4}$")


class ConfigError(ValueError):
    """Raised when a setting is invalid."""


@dataclass
class Config:
    stream_1_url: str = PLACEHOLDER_STREAM_1
    stream_2_url: str = PLACEHOLDER_STREAM_2
    event_threshold_seconds: float = 30
    window_start: str = "0600"
    window_end: str = "2000"
    # --- advanced ---
    stream_1_name: str = "Stream 1 HRBT WB Exit"
    stream_2_name: str = "Stream 2 HRBT EB Exit"
    excel_file: str = "hrbt_event_log.xlsx"
    model: str = "yolov8s.pt"
    confidence: float = 0.25
    motion_pixel_fraction: float = 0.0005
    ignore_stationary_vehicles: bool = True
    stale_stream_seconds: float = 20
    show_detection_boxes: bool = True
    min_vehicle_size: float = 0.06
    detection_image_size: int = 640

    # ------------------------------------------------------------------ helpers
    def validate(self) -> "Config":
        """Normalise and check values. Raises ConfigError with a readable message."""
        self.stream_1_url = str(self.stream_1_url).strip()
        self.stream_2_url = str(self.stream_2_url).strip()
        if not self.stream_1_url:
            raise ConfigError("Stream 1 URL cannot be empty.")
        if not self.stream_2_url:
            raise ConfigError("Stream 2 URL cannot be empty.")

        try:
            self.event_threshold_seconds = float(self.event_threshold_seconds)
        except (TypeError, ValueError):
            raise ConfigError("Event threshold must be a number of seconds.") from None
        if self.event_threshold_seconds <= 0:
            raise ConfigError("Event threshold must be greater than 0 seconds.")
        if self.event_threshold_seconds.is_integer():
            self.event_threshold_seconds = int(self.event_threshold_seconds)

        self.window_start = normalise_hhmm(self.window_start, "Window start")
        self.window_end = normalise_hhmm(self.window_end, "Window end")

        try:
            self.confidence = float(self.confidence)
            self.motion_pixel_fraction = float(self.motion_pixel_fraction)
            self.stale_stream_seconds = float(self.stale_stream_seconds)
        except (TypeError, ValueError):
            raise ConfigError("Advanced numeric settings in config.json are invalid.") from None
        if not 0 < self.confidence < 1:
            raise ConfigError("confidence must be between 0 and 1.")
        try:
            self.min_vehicle_size = float(self.min_vehicle_size)
        except (TypeError, ValueError):
            raise ConfigError("min_vehicle_size must be a number.") from None
        if not 0 <= self.min_vehicle_size < 1:
            raise ConfigError("min_vehicle_size must be between 0 and 1.")
        try:
            self.detection_image_size = int(self.detection_image_size)
        except (TypeError, ValueError):
            raise ConfigError("detection_image_size must be a whole number.") from None
        if not 160 <= self.detection_image_size <= 1920 or self.detection_image_size % 32:
            raise ConfigError("detection_image_size must be a multiple of 32 between 160 and 1920.")
        if not 0 <= self.motion_pixel_fraction < 1:
            raise ConfigError("motion_pixel_fraction must be between 0 and 1.")
        if self.stale_stream_seconds <= 0:
            raise ConfigError("stale_stream_seconds must be greater than 0.")
        self.ignore_stationary_vehicles = bool(self.ignore_stationary_vehicles)
        self.show_detection_boxes = bool(self.show_detection_boxes)
        return self

    def streams_are_placeholders(self) -> bool:
        return not (is_url(self.stream_1_url) and is_url(self.stream_2_url))

    def window_minutes(self) -> tuple[int, int]:
        return hhmm_to_minutes(self.window_start), hhmm_to_minutes(self.window_end)


def is_url(text: str) -> bool:
    return bool(re.match(r"^(https?|rtsp|rtmp)://\S+$", text.strip(), re.IGNORECASE))


def normalise_hhmm(value, label: str = "Time") -> str:
    """Accept 600, "600", "0600", "06:00" -> "0600". 2400 is allowed (end of day)."""
    text = str(value).strip().replace(":", "")
    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)
    if not _HHMM.match(text):
        raise ConfigError(f"{label} must be a 24-hour time like 0600 or 2000 (got '{value}').")
    hh, mm = int(text[:2]), int(text[2:])
    if mm > 59 or hh > 24 or (hh == 24 and mm != 0):
        raise ConfigError(f"{label} '{value}' is not a valid 24-hour time.")
    return text


def hhmm_to_minutes(text: str) -> int:
    return int(text[:2]) * 60 + int(text[2:])


def config_path() -> Path:
    return app_dir() / CONFIG_NAME


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = Path(path) if path else config_path()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=4)
        fh.write("\n")
    tmp.replace(path)


def load_config(path: Path | None = None) -> tuple[Config, bool]:
    """Load config.json, creating it with defaults if it is missing.

    Returns (config, created) where created is True if a new file was written.
    Missing keys are filled with defaults; unknown keys are ignored.
    Raises ConfigError if the file exists but can't be parsed / validated.
    """
    path = Path(path) if path else config_path()
    if not path.exists():
        cfg = Config()
        save_config(cfg, path)
        return cfg, True

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a JSON object.")

    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in data.items() if k in known})
    cfg.validate()
    if set(data) != known:  # add any newly introduced keys to the file
        save_config(cfg, path)
    return cfg, False
