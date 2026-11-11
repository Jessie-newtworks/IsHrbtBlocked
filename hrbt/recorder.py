"""Footage recording: rolling full recording, event clips and near-miss clips.

Why: to review false events, catch real blockages on video, and check for
stoppages the app *missed* (only a full recording can show those).

How
---
* Full recording. One `ffmpeg -c copy` process per camera saves the camera's
  own H.264 video, without decoding or re-encoding it (about 1 % CPU). It
  opens its own HLS connection, separate from the detector's, and writes
  short MPEG-TS segments named by their start time:

      recordings/full/stream1/stream1_2026-11-10_06-00-00.ts

  It runs during the logging window (plus the clip pre-/post-roll at either
  end). If ffmpeg exits or stops writing, it is restarted, and the problem is
  written to errors.log. Segments older than `recording_keep_days` are
  deleted, oldest first.
* Event clips. When an event ends, the segments covering it (from
  `event_clip_pre_seconds` before the start to `event_clip_post_seconds`
  after the end) are joined into one file named like the events.log line:

      recordings/events/2026-11-10_07-41-03_Stream_2_HRBT_EB_Exit_event_0h03m12s.ts

  Joining .ts segments is a plain file copy: no ffmpeg, no CPU. Clips start
  and end on segment boundaries, so they are up to one segment longer than
  asked. Clips are kept until the folder passes `event_clips_max_gb`; then
  the oldest are deleted.
* Near-miss clips (optional, `near_miss_clips`). The same for gaps with no
  passing vehicle of `near_miss_min_seconds` up to the event threshold, kept
  in recordings/near_misses/ under their own cap.
* Disk guard. Old full-recording segments are deleted early, oldest first, so
  that at least `min_free_disk_gb` stays free. If that isn't enough,
  recording pauses (logged to errors.log) until there is room again.
  Clips are never deleted to make room - only by their own cap.

.ts files play in VLC (and most players). Segments from one ffmpeg run join
seamlessly; at a reconnect the player may skip briefly.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .error_log import log_error
from .event_tracker import in_window

GB = 1024 ** 3
FULL_DIR = "full"
EVENTS_DIR = "events"
NEAR_MISS_DIR = "near_misses"
SEGMENT_EXT = ".ts"
_SEG_TIME_FMT = "%Y-%m-%d_%H-%M-%S"
_SEG_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.ts$")

TICK_S = 1.0                    # supervisor loop interval
HOUSEKEEP_EVERY_S = 60.0        # retention / cap / free-space checks
STALL_CHECK_EVERY_S = 15.0
STALL_SECONDS = 60.0            # no new data for this long -> restart ffmpeg
STABLE_AFTER_S = 120.0          # a run this long counts as recovered
RESTART_DELAY_S = 5.0           # first restart delay, doubles up to the max
RESTART_DELAY_MAX_S = 60.0
REPEAT_ERROR_EVERY = 300.0      # re-log a persistent problem at most every 5 min
PRUNE_MIN_AGE_S = 600.0         # never prune segments younger than this for space
DISK_RESUME_MARGIN = 1 * GB     # free space above the floor needed to resume
CLIP_MARGIN_S = 10.0            # wait this long after a clip's footage should exist


# ===================================================================== helpers
def safe_name(text: str) -> str:
    """Stream name -> file-name friendly text ("Stream 1 HRBT WB Exit" ->
    "Stream_1_HRBT_WB_Exit")."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", text).strip("_") or "stream"


def stream_key(index: int) -> str:
    return f"stream{index}"


def segment_template(folder: Path, key: str) -> str:
    """ffmpeg strftime output pattern for one camera's segments."""
    escaped = str(folder).replace("%", "%%")           # a literal % in the folder name
    return os.path.join(escaped, f"{key}_{_SEG_TIME_FMT}{SEGMENT_EXT}")


def parse_segment_time(path: Path) -> float | None:
    """Start time (epoch seconds, local clock) from a segment's file name."""
    m = _SEG_RE.search(Path(path).name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), _SEG_TIME_FMT).timestamp()
    except ValueError:
        return None


@dataclass
class Segment:
    path: Path
    start: float     # from the file name
    end: float       # last write (file mtime)
    size: int


def list_segments(folder: Path) -> list[Segment]:
    """All recognised segments in a folder, oldest first."""
    out = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return out
    for e in entries:
        start = parse_segment_time(Path(e.name)) if e.is_file() else None
        if start is None:
            continue
        try:
            st = e.stat()
        except OSError:
            continue
        out.append(Segment(Path(e.path), start, max(start, st.st_mtime), st.st_size))
    out.sort(key=lambda s: s.start)
    return out


def select_segments(segments: list[Segment], t0: float, t1: float) -> list[Segment]:
    """Segments that overlap [t0, t1], oldest first."""
    return [s for s in segments if s.size > 0 and s.start <= t1 and s.end >= t0]


def recording_active(now: datetime, start_min: int, end_min: int,
                     pre_s: float = 0.0, post_s: float = 0.0) -> bool:
    """True inside the logging window widened by pre_s before its start and
    post_s after its end (so clip pre-/post-roll at the edges is recorded)."""
    return (in_window(now, start_min, end_min)
            or in_window(now + timedelta(seconds=pre_s), start_min, end_min)
            or in_window(now - timedelta(seconds=post_s), start_min, end_min))


def clip_filename(stream_name: str, kind: str, start_ts: float, seconds: float) -> str:
    """Name that matches the events.log line, e.g.
    2026-11-10_07-41-03_Stream_2_HRBT_EB_Exit_event_0h03m12s.ts"""
    stamp = datetime.fromtimestamp(start_ts).strftime(_SEG_TIME_FMT)
    s = int(round(seconds))
    if kind == "event":
        length = f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"
    else:
        length = f"{s}s"
    return f"{stamp}_{safe_name(stream_name)}_{kind}_{length}{SEGMENT_EXT}"


def join_segments(segments: list[Segment], dest: Path) -> int:
    """Join MPEG-TS segments into one file (a plain byte copy). Returns its size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with open(tmp, "wb") as out:
            for seg in segments:
                with open(seg.path, "rb") as fh:
                    shutil.copyfileobj(fh, out, 1024 * 1024)
        tmp.replace(dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return dest.stat().st_size


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def _files_oldest_first(folder: Path, suffix: str = SEGMENT_EXT) -> list[tuple[str, Path, int]]:
    out = []
    try:
        for e in os.scandir(folder):
            if e.is_file() and e.name.endswith(suffix):
                try:
                    out.append((e.name, Path(e.path), e.stat().st_size))
                except OSError:
                    pass
    except OSError:
        pass
    out.sort()     # names start with the date and time
    return out


def enforce_cap(folder: Path, max_bytes: int) -> list[Path]:
    """Delete the oldest clips until the folder is within max_bytes. The newest
    clip is always kept. Returns the deleted paths."""
    files = _files_oldest_first(folder)
    total = sum(size for _, _, size in files)
    deleted = []
    for _, path, size in files[:-1]:
        if total <= max_bytes:
            break
        try:
            path.unlink()
            total -= size
            deleted.append(path)
        except OSError as exc:
            log_error(f"Recording: could not delete old clip {path.name}.", exc)
    return deleted


def prune_old_segments(full_dir: Path, older_than: float) -> int:
    """Delete full-recording segments that started before `older_than` (epoch s).
    Returns the number deleted."""
    n = 0
    try:
        subdirs = [Path(e.path) for e in os.scandir(full_dir) if e.is_dir()]
    except OSError:
        return 0
    for d in subdirs:
        for seg in list_segments(d):
            if seg.start >= older_than:
                break
            try:
                seg.path.unlink()
                n += 1
            except OSError as exc:
                log_error(f"Recording: could not delete old segment {seg.path.name}.", exc)
    return n


def build_ffmpeg_cmd(ffmpeg: str, url: str, folder: Path, key: str,
                     segment_seconds: int) -> list[str]:
    """ffmpeg command that saves the stream unchanged in time-named segments."""
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", url,
        "-map", "0:v:0", "-c", "copy",              # video only, no re-encoding
        "-f", "segment",
        "-segment_time", str(int(segment_seconds)),
        "-segment_atclocktime", "1",                 # cut on the clock: 06:00, 06:05 ...
        "-segment_format", "mpegts",
        "-strftime", "1",
        segment_template(folder, key),
    ]


def kill_orphan_recorders(folder: Path) -> int:
    """Stop ffmpeg processes left recording into `folder` by a copy of the app
    that crashed (Linux only; run_hrbt.sh restarts the app after a crash, and
    the old ffmpeg would otherwise keep recording a second copy). Returns the
    number stopped."""
    if not sys.platform.startswith("linux"):
        return 0
    import signal
    marker = str(folder)
    n = 0
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            if int(entry.name) == os.getpid():
                continue
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        if args and Path(args[0].decode(errors="replace")).name == "ffmpeg" and \
                any(marker in a.decode(errors="replace") for a in args):
            try:
                os.kill(int(entry.name), signal.SIGTERM)
                n += 1
            except OSError:
                pass
    return n


# ============================================================ one camera
class SegmentRecorder:
    """Keeps one ffmpeg process recording one camera. Driven by
    FootageRecorder's supervisor thread through supervise()."""

    def __init__(self, index: int, name: str, url: str, folder: Path, ffmpeg: str,
                 segment_seconds: int):
        self.index = index
        self.name = name
        self.url = url
        self.key = stream_key(index)
        self.folder = Path(folder)
        self.ffmpeg = ffmpeg
        self.segment_seconds = int(segment_seconds)
        self.proc: subprocess.Popen | None = None
        self.stderr_path = self.folder.parent / f"ffmpeg_{self.key}.log"
        self._started = 0.0
        self._retry_at = 0.0
        self._delay = RESTART_DELAY_S
        self._failing = False
        self._last_error_log = 0.0
        self._next_stall_check = 0.0
        self._last_size = -1
        self._last_growth = 0.0

    @property
    def recording(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def supervise(self, now: float, want: bool) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            self._on_exit(now)
        if not want:
            if self.proc is not None:
                self.stop()
            return
        if self.proc is None:
            if now >= self._retry_at:
                self._spawn(now)
            return
        if now >= self._next_stall_check:
            self._next_stall_check = now + STALL_CHECK_EVERY_S
            self._check_stall(now)
        if self._failing and now - self._started >= STABLE_AFTER_S and \
                now - self._last_growth < STALL_SECONDS:
            log_error(f"{self.name}: footage recording is running again.", level="INFO")
            self._failing = False
            self._delay = RESTART_DELAY_S

    def _spawn(self, now: float) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_cmd(self.ffmpeg, self.url, self.folder, self.key,
                               self.segment_seconds)
        kwargs = {}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            with open(self.stderr_path, "wb") as err:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=err, **kwargs)
        except OSError as exc:
            self.proc = None
            self._problem(now, f"{self.name}: could not start ffmpeg for footage recording.", exc)
            return
        self._started = self._last_growth = now
        self._last_size = -1
        self._next_stall_check = now + STALL_CHECK_EVERY_S

    def _on_exit(self, now: float) -> None:
        code = self.proc.returncode
        self.proc = None
        detail = self._stderr_tail()
        self._problem(now, f"{self.name}: footage recording (ffmpeg) stopped, exit code "
                           f"{code}{': ' + detail if detail else ''}. Restarting.")

    def _check_stall(self, now: float) -> None:
        segs = list_segments(self.folder)
        newest = segs[-1] if segs else None
        size = (newest.size, str(newest.path)) if newest else None
        if size != self._last_size:
            self._last_size = size
            self._last_growth = now
        elif now - self._last_growth >= STALL_SECONDS:
            self._problem(now, f"{self.name}: footage recording stalled (no new video for "
                               f"{STALL_SECONDS:.0f} s). Restarting ffmpeg.")
            self.stop()

    def _problem(self, now: float, message: str, exc: BaseException | None = None) -> None:
        if not self._failing or now - self._last_error_log > REPEAT_ERROR_EVERY:
            log_error(message, exc)
            self._last_error_log = now
        self._failing = True
        self._retry_at = now + self._delay
        self._delay = min(RESTART_DELAY_MAX_S, self._delay * 2)

    def _stderr_tail(self, lines: int = 3) -> str:
        try:
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return " | ".join(line.strip() for line in text.strip().splitlines()[-lines:])

    def stop(self, timeout: float = 5.0) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(2)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass


# ====================================================== both cameras + clips
@dataclass
class ClipJob:
    index: int
    kind: str          # "event" or "gap"
    start: float       # when the event / gap started (stream time, epoch s)
    end: float
    seconds: float     # length shown in the file name
    t0: float          # footage wanted from ...
    t1: float          # ... to


class FootageRecorder:
    """Records both cameras and cuts clips. Start with start(); stop() ends
    the recordings and writes any clips still waiting."""

    def __init__(self, cfg, base_dir: Path):
        self.cfg = cfg
        folder = Path(cfg.recording_folder)
        self.root = folder if folder.is_absolute() else Path(base_dir) / folder
        self.full_dir = self.root / FULL_DIR
        self.events_dir = self.root / EVENTS_DIR
        self.near_dir = self.root / NEAR_MISS_DIR
        self.window = cfg.window_minutes()
        self.names = {1: cfg.stream_1_name, 2: cfg.stream_2_name}
        self.pre = float(cfg.event_clip_pre_seconds)
        self.post = float(cfg.event_clip_post_seconds)
        self.keep_seconds = float(cfg.recording_keep_days) * 86400
        self.floor = int(float(cfg.min_free_disk_gb) * GB)
        self.ffmpeg = shutil.which("ffmpeg")
        self.recorders: dict[int, SegmentRecorder] = {}
        self.disk_low = False
        self._jobs: list[ClipJob] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_housekeeping = 0.0
        self._last_disk_log = 0.0
        if self.ffmpeg:
            for i, url in ((1, cfg.stream_1_url), (2, cfg.stream_2_url)):
                self.recorders[i] = SegmentRecorder(
                    i, self.names[i], url, self.full_dir / stream_key(i), self.ffmpeg,
                    cfg.recording_segment_seconds)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.ffmpeg:
            log_error("Footage recording is OFF: ffmpeg was not found. On Debian install it "
                      "with:  sudo apt install -y ffmpeg   (on Windows, install ffmpeg and add "
                      "it to PATH), then restart the app. Monitoring carries on without it.")
            return
        try:
            for d in (self.full_dir, self.events_dir):
                d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_error(f"Footage recording is OFF: could not create {self.root}.", exc)
            self.ffmpeg = None
            self.recorders = {}
            return
        stray = kill_orphan_recorders(self.full_dir)
        if stray:
            log_error(f"Recording: stopped {stray} ffmpeg process(es) left running by an "
                      "earlier run of the app.", level="INFO")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop ffmpeg and write the clips still waiting (with what was recorded)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(10)
            self._thread = None
        for rec in self.recorders.values():
            rec.stop()
        with self._lock:
            jobs, self._jobs = self._jobs, []
        for job in jobs:
            self._make_clip(job)

    @property
    def status(self) -> str:
        if not self.ffmpeg:
            return "Footage: OFF (ffmpeg not found - see errors.log)"
        if self.disk_low:
            return "Footage: PAUSED - disk nearly full"
        if any(r.recording for r in self.recorders.values()):
            return f"Footage: recording to {self.root.name}/"
        return "Footage: waiting for the logging window"

    # ------------------------------------------------------------------ clips
    def add_clip(self, index: int, kind: str, start: float, end: float,
                 seconds: float) -> None:
        """Ask for a clip (thread-safe; called by the detector threads).
        kind: "event" or "gap" (near miss)."""
        if not self.ffmpeg:
            return
        if kind == "gap" and not self.cfg.near_miss_clips:
            return
        job = ClipJob(index, kind, start, end, seconds, start - self.pre, end + self.post)
        with self._lock:
            self._jobs.append(job)

    def _clip_due(self, job: ClipJob, now: float) -> bool:
        if now < job.t1 + CLIP_MARGIN_S:
            return False
        rec = self.recorders.get(job.index)
        if rec is None or not rec.recording:
            return True                      # nothing more is coming
        if now > job.t1 + 2 * rec.segment_seconds + 60:
            return True                      # don't wait forever
        # wait until the segment holding t1 is finished (a newer one has started)
        segs = list_segments(rec.folder)
        return bool(segs) and segs[-1].start > job.t1

    def _process_clips(self, now: float) -> None:
        with self._lock:
            due = [j for j in self._jobs if self._clip_due(j, now)]
            self._jobs = [j for j in self._jobs if j not in due]
        for job in due:
            self._make_clip(job)

    def _make_clip(self, job: ClipJob) -> None:
        rec = self.recorders.get(job.index)
        if rec is None:
            return
        name = self.names[job.index]
        what = "event" if job.kind == "event" else "near-miss gap"
        segs = select_segments(list_segments(rec.folder), job.t0, job.t1)
        if not segs:
            log_error(f"{name}: no footage recorded for the {what} at "
                      f"{datetime.fromtimestamp(job.start):%Y-%m-%d %H:%M:%S}, so no clip.",
                      level="WARNING")
            return
        folder = self.events_dir if job.kind == "event" else self.near_dir
        dest = folder / clip_filename(name, job.kind, job.start, job.seconds)
        needed = sum(s.size for s in segs)
        try:
            if free_bytes(self.root) - needed < self.floor:
                self._free_space(time.time(), needed)
            if free_bytes(self.root) - needed < self.floor:
                log_error(f"{name}: not enough free disk space for the {what} clip "
                          f"{dest.name} - skipped.")
                return
            join_segments(segs, dest)
        except OSError as exc:
            log_error(f"{name}: could not save clip {dest.name}.", exc)
            return
        cap = self.cfg.event_clips_max_gb if job.kind == "event" else \
            self.cfg.near_miss_clips_max_gb
        enforce_cap(folder, int(float(cap) * GB))

    # ----------------------------------------------------------- supervisor
    def _want_recording(self, index: int, now: float) -> bool:
        if self.disk_low:
            return False
        if recording_active(datetime.fromtimestamp(now), *self.window, self.pre, self.post):
            return True
        with self._lock:                     # finish post-roll for a late clip
            return any(j.index == index and j.t1 > now for j in self._jobs)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            try:
                if now >= self._next_housekeeping:
                    self._next_housekeeping = now + HOUSEKEEP_EVERY_S
                    self.housekeeping(now)
                for i, rec in self.recorders.items():
                    rec.supervise(now, self._want_recording(i, now))
                self._process_clips(now)
            except Exception as exc:
                log_error("Footage recorder error.", exc)
            self._stop.wait(TICK_S)

    def housekeeping(self, now: float) -> None:
        """Retention, free-space guard."""
        prune_old_segments(self.full_dir, now - self.keep_seconds)
        try:
            free = free_bytes(self.root)
        except OSError as exc:
            log_error("Recording: could not check free disk space.", exc)
            return
        if free < self.floor:
            free = self._free_space(now, 0)
        if free < self.floor:
            if not self.disk_low or now - self._last_disk_log > REPEAT_ERROR_EVERY:
                log_error(f"Footage recording PAUSED: only {free / GB:.1f} GB free on the disk "
                          f"(min_free_disk_gb is {self.floor / GB:.1f}), even after deleting "
                          f"old recordings.")
                self._last_disk_log = now
            if not self.disk_low:
                self.disk_low = True
                for rec in self.recorders.values():
                    rec.stop()
        elif self.disk_low and free >= self.floor + DISK_RESUME_MARGIN:
            self.disk_low = False
            log_error(f"Footage recording resumed: {free / GB:.1f} GB free.", level="INFO")

    def _free_space(self, now: float, extra: int) -> int:
        """Delete the oldest full-recording segments (not the last 10 min, and
        not footage a waiting clip needs) until `extra` bytes fit above the
        floor. Returns the free space afterwards."""
        with self._lock:
            needed = [(j.index, j.t0, j.t1) for j in self._jobs]
        segs = []
        for i, rec in self.recorders.items():
            for s in list_segments(rec.folder):
                if s.start > now - PRUNE_MIN_AGE_S:
                    continue
                if any(i == ji and s.start <= t1 and s.end >= t0 for ji, t0, t1 in needed):
                    continue
                segs.append(s)
        segs.sort(key=lambda s: s.start)
        free = free_bytes(self.root)
        for s in segs:
            if free - extra >= self.floor:
                break
            try:
                s.path.unlink()
                free += s.size
            except OSError as exc:
                log_error(f"Recording: could not delete segment {s.path.name}.", exc)
        return free_bytes(self.root)
