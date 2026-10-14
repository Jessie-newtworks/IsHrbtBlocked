"""Live stream reader + YOLOv8 vehicle detector for one camera.

Each StreamMonitor runs two background threads:

* reader   - keeps the stream open with OpenCV/FFmpeg, puts *every* decoded
             frame (with its stream timestamp) into a queue for the detector,
             keeps the newest frame for the live view, and reconnects
             automatically if the stream drops.
* detector - processes every queued frame in order. The cameras run at about
             15 fps or less, so a fast car may be in view for only 2-3 frames.
             Checking every frame means it can't slip through.

To keep up on a CPU, the detector first does a cheap motion check: a frame
that barely differs from the previous one can't contain a passing car, so YOLO
is skipped for it. While the exit is empty (the case that matters for an
event) almost no YOLO runs are needed. Set motion_pixel_fraction to 0 to run
YOLO on every frame.

"Passing": the cameras watch the tunnel exits, so when traffic is stopped
inside / before the tunnel no cars come out. With ignore_stationary_vehicles=true
(default) a vehicle only counts as stopped if it stayed in place at every
check over the last 2 seconds (0.5, 1, 1.5 and 2 s back). Otherwise it counts
as moving, so a parked or disabled vehicle in view
can't hide a stoppage. A vehicle seen in a single frame with nothing at that
spot before counts as passing. With the setting false, any vehicle in view
counts.

Live view: frames are shown at the pace of their stream timestamps, through
a small playback buffer, the way a browser plays HLS. Showing each frame the
moment it was decoded made the view race through each chunk and then freeze
until the next one arrived. Detection boxes are matched to the frame on
screen.

Timing uses the stream's own frame timestamps, not the moment each frame
reached the app. HLS delivers video in bursts of a few seconds at a time, and
using arrival time would distort the gaps between cars. A gap of more than
MAX_FRAME_GAP_S in the video itself (dropped segments) counts as "no picture",
never as an empty road.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .error_log import log_error, log_event
from .event_tracker import EventTracker

# COCO class ids that count as vehicles: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]
# "Same place" test, tolerant of YOLO's box wobble. Two boxes are in the same
# place if their centres are within max(JITTER_PX, CENTRE_TOL x box size) and
# their sizes differ by less than SIZE_TOL. (An overlap/IoU test failed on
# small boxes: a 2-pixel wobble on a 20-pixel box looked like movement, so a
# false detection deep in the tunnel counted as a passing car during a real
# blockage.)
JITTER_PX = 4.0
CENTRE_TOL = 0.25
SIZE_TOL = 1.2
# A vehicle counts as STOPPED only if it was in the same place at every one of
# these look-back points. At 320x240 a distant car, or one
# driving straight toward or away from the camera, barely changes in 0.5 s,
# but it clearly does over 2 s. Checking several points, not just 2 s back,
# stops the next car in line (often ~2 s behind) from making a moving car
# look parked.
MOTION_LOOKBACKS_S = (0.5, 1.0, 1.5, 2.0)
MOTION_SIZE = (320, 180)       # frame size used for the cheap motion check
MOTION_PIXEL_DELTA = 25        # grey-level change that counts as a "changed" pixel
QUEUE_SECONDS = 20             # frames buffered for the detector (at 15 fps)
# A gap this long between consecutive frames is missing video, not an empty
# road: it is reported as "no picture" and never counted as blocked time.
MAX_FRAME_GAP_S = 3.0
PLAYBACK_MARGIN_S = 0.3        # extra playback delay on top of the measured jitter
LAG_WINDOW_S = 30.0            # how far back to look when sizing the playback delay
# The cameras only take 2-4 real pictures per second, so the time since the
# last car sighting restarts between every picture. Keep showing "Traffic
# moving" until the road has been clear this long (display only - events
# still use the full threshold).
BANNER_CLEAR_S = 2.0
PERF_FIRST_REPORT_S = 120.0    # first CPU-load line in errors.log after 2 min
PERF_REPORT_EVERY_S = 1800.0   # then every 30 min
RECONNECT_DELAY = 5.0          # seconds between reconnect attempts
REPEAT_ERROR_EVERY = 300.0     # re-log a persistent problem at most every 5 min

_model_lock = threading.Lock()  # serialise model loading / first download


def load_model(model_path: Path):
    """Load YOLO weights (ultralytics downloads yolov8s.pt on first use)."""
    with _model_lock:
        _limit_cpu_threads()
        from ultralytics import YOLO  # slow import - keep it off the GUI thread
        return YOLO(str(model_path))


_threads_limited = False


def _limit_cpu_threads() -> None:
    """By default PyTorch grabs every CPU core for each inference. With two
    detectors running, that starves video decoding and the GUI, so the
    streams stutter. Give each detector about half the cores, minus one."""
    global _threads_limited
    if _threads_limited:
        return
    _threads_limited = True
    try:
        import torch
        cores = os.cpu_count() or 2
        torch.set_num_threads(max(1, cores // 2 - 1))
    except Exception:
        pass


def box_size(b) -> float:
    return max(1e-6, ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5)


def same_place(a, b) -> bool:
    """True if box a is where box b was, allowing for detector wobble."""
    sa, sb = box_size(a), box_size(b)
    if abs(sa - sb) > max(JITTER_PX, (SIZE_TOL - 1) * min(sa, sb)):
        return False                            # grew / shrank: approaching or leaving
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return (dx * dx + dy * dy) ** 0.5 <= max(JITTER_PX, CENTRE_TOL * min(sa, sb))


def moving_flags(boxes: list, references: list[list] | None) -> list[bool]:
    """For each box: False (stopped) only if every reference detection set
    contains a box in the same place; True (moving) otherwise."""
    if not references:
        return [True] * len(boxes)  # not enough history yet - treat as moving
    return [not all(any(same_place(b, p) for p in ref) for ref in references)
            for b in boxes]


class BoxHistory:
    """Recent YOLO results, used to look up the detections from 0.5-2 s ago."""

    def __init__(self, lookbacks: tuple[float, ...] = MOTION_LOOKBACKS_S):
        self.lookbacks = tuple(sorted(lookbacks))
        self._items: deque[tuple[float, list]] = deque()

    def clear(self) -> None:
        self._items.clear()

    def _index_at_or_before(self, t: float) -> int | None:
        for i in range(len(self._items) - 1, -1, -1):
            if self._items[i][0] <= t:
                return i
        return None

    def references(self, ts: float) -> list[list] | None:
        """Detections at each look-back point (newest at least that old).
        None until the history covers the longest look-back."""
        indexes = [self._index_at_or_before(ts - lb) for lb in self.lookbacks]
        if any(i is None for i in indexes):
            return None
        refs = [self._items[i][1] for i in indexes]
        for _ in range(indexes[-1]):        # older entries are no longer needed
            self._items.popleft()
        return refs

    def add(self, ts: float, boxes: list) -> None:
        self._items.append((ts, boxes))


class StreamClock:
    """Turns the stream's presentation timestamps (ms) into wall-clock seconds.

    The clock is anchored on the first frame. After that, time follows the
    video's own timestamps, so HLS bursts (several seconds of video decoded
    at once) and network stalls don't stretch or squash the gaps between
    cars. If the timestamps jump backwards or far forwards (a stream restart),
    it re-anchors.
    """

    RESYNC_JUMP_S = 60.0

    def __init__(self):
        self._offset: float | None = None
        self._last_pts: float | None = None
        self._last = 0.0

    def stamp(self, pts_ms: float, arrival: float) -> float:
        pts = None if pts_ms is None or not math.isfinite(pts_ms) or pts_ms < 0 \
            else pts_ms / 1000.0
        if pts is None or pts == self._last_pts:
            ts = arrival                               # no usable / advancing PTS
        else:
            jumped = self._last_pts is not None and not (
                0 <= pts - self._last_pts <= self.RESYNC_JUMP_S)
            if self._offset is None or jumped:
                self._offset = arrival - pts
            self._last_pts = pts
            ts = self._offset + pts
        ts = max(ts, self._last + 1e-4)                # keep strictly increasing
        self._last = ts
        return ts


class _StreamStats:
    """Measures a stream's real frame rate and delivery pattern for 30 s after
    each connect, then writes one INFO line to errors.log."""

    DURATION_S = 30.0

    def __init__(self, name: str):
        self.name = name
        self.first_arrival = None
        self.first_ts = None
        self.last_arrival = None
        self.frames = 0
        self.new_images = 0
        self.prev_small = None
        self.chunks = 1
        self.done = False

    def add(self, arrival: float, ts: float, frame: np.ndarray) -> None:
        if self.done:
            return
        if self.first_arrival is None:
            self.first_arrival, self.first_ts = arrival, ts
        elif arrival - self.last_arrival > 0.5:
            self.chunks += 1                       # a pause between deliveries
        self.last_arrival = arrival
        self.frames += 1
        small = cv2.cvtColor(cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.int16)
        if self.prev_small is None or np.abs(small - self.prev_small).mean() > 0.3:
            self.new_images += 1                   # not a repeat of the previous frame
        self.prev_small = small
        if arrival - self.first_arrival >= self.DURATION_S and ts > self.first_ts:
            self.done = True
            fps = (self.frames - 1) / (ts - self.first_ts)
            h, w = frame.shape[:2]
            elapsed = arrival - self.first_arrival
            new_per_s = (self.new_images - 1) / (ts - self.first_ts)
            log_error(f"{self.name}: {w}x{h}, about {fps:.1f} fps, of which about "
                      f"{new_per_s:.1f} new pictures per second (the rest are repeats); "
                      f"video arrives in chunks about every {elapsed / self.chunks:.1f} s.",
                      level="INFO")


class StreamMonitor:
    def __init__(self, index: int, name: str, url: str, tracker: EventTracker,
                 model_path: Path, confidence: float, motion_pixel_fraction: float,
                 ignore_stationary: bool, stale_seconds: float,
                 min_vehicle_size: float = 0.06, image_size: int = 640):
        self.index = index
        self.min_vehicle_size = max(0.0, float(min_vehicle_size))
        self.image_size = int(image_size)
        self.name = name
        self.url = url
        self.tracker = tracker
        self.model_path = model_path
        self.confidence = confidence
        self.motion_fraction = max(0.0, float(motion_pixel_fraction))
        self.ignore_stationary = ignore_stationary
        self.stale_seconds = max(5.0, float(stale_seconds))

        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None          # frame currently on screen
        self._shown_ts: float | None = None
        # playback buffer: (stream ts, frame), shown when ts <= now - self._lag
        self._display: deque[tuple[float, np.ndarray]] = deque(maxlen=QUEUE_SECONDS * 30)
        self._lag = 0.0
        self._lag_samples: deque[tuple[float, float]] = deque()
        # detections per frame timestamp: (ts, [(box, moving), ...])
        self._box_log: deque[tuple[float, list]] = deque(maxlen=QUEUE_SECONDS * 30)
        # tracker state per frame timestamp, so the banner matches the picture
        self._status_log: deque[tuple[float, object]] = deque(maxlen=QUEUE_SECONDS * 30)
        self._queue: deque[tuple[np.ndarray, float]] = deque(maxlen=QUEUE_SECONDS * 15)
        self._queue_cv = threading.Condition()
        self._dropped = 0
        self._last_frame_wall = time.monotonic()
        self.status = "Connecting..."
        self.model_ready = False
        self.behind = False
        self._threads: list[threading.Thread] = []
        self._perf_runs = 0
        self._perf_busy = 0.0
        self._perf_since = time.monotonic()
        self._perf_next = time.monotonic() + PERF_FIRST_REPORT_S
        self._event_start: float | None = None
        self._event_secs = 0.0

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader_loop, name=f"reader-{self.index}", daemon=True),
            threading.Thread(target=self._detector_loop, name=f"detector-{self.index}", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal both threads to stop. They are daemons, so a reader blocked
        inside FFmpeg can never keep the app from closing."""
        self._stop.set()
        with self._queue_cv:
            self._queue_cv.notify_all()
        if self._event_start is not None:
            log_event(f"{self.name}: logging STOPPED during an event - blocked "
                      f"{fmt_duration(self._event_secs)} so far")
            self._event_start, self._event_secs = None, 0.0
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))
        self._clear_display()
        self._queue.clear()

    def _clear_display(self) -> None:
        with self._frame_lock:
            self._frame = None
            self._shown_ts = None
            self._display.clear()
            self._box_log.clear()
            self._status_log.clear()
        self._lag_samples.clear()
        self._lag = 0.0

    # ------------------------------------------------------------------ reader
    def _open_capture(self) -> cv2.VideoCapture:
        params = []
        for prop, value in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 15000),
                            ("CAP_PROP_READ_TIMEOUT_MSEC", 15000)):
            if hasattr(cv2, prop):
                params += [getattr(cv2, prop), value]
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        except TypeError:  # very old OpenCV without the params overload
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return cap

    def _push(self, frame: np.ndarray, ts: float) -> None:
        with self._queue_cv:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1                    # detector is behind
            self._queue.append((frame, ts))
            self._queue_cv.notify()

    def _reader_loop(self) -> None:
        last_error_log = 0.0
        failing = False
        while not self._stop.is_set():
            cap = self._open_capture()
            if not cap.isOpened():
                cap.release()
                self.status = "No signal - retrying..."
                now = time.monotonic()
                if not failing or now - last_error_log > REPEAT_ERROR_EVERY:
                    log_error(f"{self.name}: could not open stream {self.url}")
                    last_error_log, failing = now, True
                self._stop.wait(RECONNECT_DELAY)
                continue

            if failing:
                log_error(f"{self.name}: stream reconnected.", level="INFO")
            failing = False
            self.status = "Live"
            clock = StreamClock()
            self._clear_display()
            stats = _StreamStats(self.name)
            bad_reads = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    bad_reads += 1
                    if bad_reads >= 50:
                        break
                    time.sleep(0.05)
                    continue
                bad_reads = 0
                arrival = time.time()
                ts = clock.stamp(cap.get(cv2.CAP_PROP_POS_MSEC), arrival)
                self._update_lag(arrival, ts)
                stats.add(arrival, ts, frame)
                with self._frame_lock:
                    self._display.append((ts, frame))
                self._last_frame_wall = time.monotonic()
                self._push(frame, ts)
            cap.release()

            if not self._stop.is_set():
                self.status = "Stream dropped - reconnecting..."
                log_error(f"{self.name}: stream stopped delivering frames; reconnecting.")
                last_error_log, failing = time.monotonic(), True
                self._stop.wait(RECONNECT_DELAY)

    # ---------------------------------------------------------------- detector
    def _next_frame(self, wait: float):
        with self._queue_cv:
            if not self._queue and not self._stop.is_set():
                self._queue_cv.wait(wait)
            if self._queue:
                return self._queue.popleft()
        return None

    def _detector_loop(self) -> None:
        try:
            model = load_model(self.model_path)
            self.model_ready = True
            with self._queue_cv:            # discard frames that piled up while loading
                self._queue.clear()
                self._dropped = 0
        except Exception as exc:
            log_error(f"{self.name}: could not load YOLO model '{self.model_path}'.", exc)
            self.status = "ERROR: model failed to load (see errors.log)"
            return

        history = BoxHistory()
        prev_small: np.ndarray | None = None
        last_flagged: list = []      # [(box, moving)] from the last YOLO run
        last_drop_log = 0.0
        prev_ts: float | None = None
        down = False

        while not self._stop.is_set():
            item = self._next_frame(wait=1.0)

            # ---- report a dead / stalled feed (no frames for stale_seconds) ----
            if item is None:
                if self._stop.is_set():
                    break
                if time.monotonic() - self._last_frame_wall > self.stale_seconds:
                    self.tracker.observe(datetime.now(), None)
                    if not down:
                        down = True
                        prev_small = None
                        prev_ts = None
                        history.clear()
                        self._clear_display()
                        if self.status == "Live":
                            self.status = "No frames - waiting for stream..."
                continue
            if down:
                down = False
                if self.status.startswith("No frames"):
                    self.status = "Live"

            frame, ts = item
            if prev_ts is not None and ts - prev_ts > MAX_FRAME_GAP_S:
                self.tracker.observe(datetime.fromtimestamp(prev_ts + 1e-3), None)
                prev_small = None
                history.clear()
            prev_ts = ts
            # ---- detector falling behind? (frames dropped from the queue) ----
            if self._dropped:
                self.behind = True
                if time.monotonic() - last_drop_log > REPEAT_ERROR_EVERY:
                    log_error(f"{self.name}: detector can't keep up - {self._dropped} frame(s) "
                              "skipped. A GPU, or raising motion_pixel_fraction, will help.",
                              level="WARNING")
                    last_drop_log = time.monotonic()
                self._dropped = 0
            elif not self._queue:
                self.behind = False

            # ---- cheap motion gate ----
            small = cv2.GaussianBlur(
                cv2.cvtColor(cv2.resize(frame, MOTION_SIZE, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if self.motion_fraction > 0 and prev_small is not None:
                changed = np.count_nonzero(cv2.absdiff(small, prev_small) > MOTION_PIXEL_DELTA)
                has_motion = changed >= self.motion_fraction * small.size
            else:
                has_motion = True
            prev_small = small

            if has_motion:
                try:
                    passing, last_flagged = self._detect(model, frame, ts, history)
                except Exception as exc:
                    log_error(f"{self.name}: detection failed.", exc)
                    self.tracker.observe(datetime.fromtimestamp(ts), None)
                    continue
            else:
                # Same picture as before (the cameras repeat frames between real
                # updates): nothing new passed. Keep showing the last judgement.
                passing = False
                with self._frame_lock:
                    self._box_log.append((ts, last_flagged))

            self.tracker.observe(datetime.fromtimestamp(ts), passing)
            snap = self.tracker.snapshot()
            with self._frame_lock:
                self._status_log.append((ts, snap))
            self._log_event_change(snap, ts)

    def _record_perf(self, seconds: float) -> None:
        """Log how hard YOLO is working, so the CPU needs can be judged on the
        real machine (e.g. before trying detection_image_size 960)."""
        self._perf_runs += 1
        self._perf_busy += seconds
        now = time.monotonic()
        if now < self._perf_next:
            return
        span = max(1e-6, now - self._perf_since)
        avg_ms = 1000 * self._perf_busy / self._perf_runs
        log_error(f"{self.name}: YOLO at {self.image_size} px ran {self._perf_runs / span:.1f} "
                  f"times/s, {avg_ms:.0f} ms each - detector busy {100 * self._perf_busy / span:.0f}% "
                  f"of the time{' (FALLING BEHIND)' if self.behind else ''}.", level="INFO")
        self._perf_runs, self._perf_busy, self._perf_since = 0, 0.0, now
        self._perf_next = now + PERF_REPORT_EVERY_S

    def _log_event_change(self, snap, ts: float) -> None:
        """Write each event's start and end to events.log (for checking the app
        against what really happened)."""
        if snap.in_event and self._event_start is None:
            self._event_start = ts
            log_event(f"{self.name}: event STARTED at {_clock(ts)}")
        elif snap.in_event:
            self._event_secs = snap.current_event_seconds
        elif self._event_start is not None:
            log_event(f"{self.name}: event ENDED at {_clock(ts)} - blocked "
                      f"{fmt_duration(self._event_secs)} ({self._event_secs:.0f} s)")
            self._event_start, self._event_secs = None, 0.0

    def _detect(self, model, frame, ts: float, history: BoxHistory):
        t0 = time.perf_counter()
        result = model.predict(frame, conf=self.confidence, classes=VEHICLE_CLASSES,
                               imgsz=self.image_size, verbose=False)[0]
        self._record_perf(time.perf_counter() - t0)
        found = [tuple(map(float, b)) for b in result.boxes.xyxy.tolist()] if result.boxes else []
        # Far-away detections are unreliable at this resolution (false hits,
        # wobbly boxes). Every car leaving the tunnel passes close to the
        # camera anyway, so only count vehicles at least min_vehicle_size of the
        # picture height. The others are drawn grey and ignored.
        # Movement is judged against ALL recent detections (small ones too), so
        # an object hovering around the size cut-off isn't mistaken for a new car.
        min_px = self.min_vehicle_size * frame.shape[0]
        moving = moving_flags(found, history.references(ts))
        history.add(ts, found)
        flagged = [(b, mv if box_size(b) >= min_px else None) for b, mv in zip(found, moving)]
        with self._frame_lock:
            self._box_log.append((ts, flagged))
        counted = [mv for _, mv in flagged if mv is not None]
        passing = any(counted) if self.ignore_stationary else bool(counted)
        return passing, flagged

    # ----------------------------------------------------------------- display
    def _update_lag(self, arrival: float, ts: float) -> None:
        """Size the playback delay so the buffer never runs dry.

        arrival - ts is how late a frame showed up compared with its place in
        the video. The delay is set to the worst case seen in the last 30 s,
        plus a margin. It goes up immediately and comes down slowly, so the
        picture doesn't jump.
        """
        self._lag_samples.append((arrival, arrival - ts))
        while self._lag_samples and self._lag_samples[0][0] < arrival - LAG_WINDOW_S:
            self._lag_samples.popleft()
        target = min(15.0, max(0.0, max(x for _, x in self._lag_samples) + PLAYBACK_MARGIN_S))
        if target > self._lag:
            self._lag = target
        else:
            self._lag = max(target, self._lag - 0.005)

    def _advance_playback(self, now: float) -> None:
        """Move to the newest buffered frame whose time has come (call with lock held)."""
        playhead = now - self._lag
        while self._display and self._display[0][0] <= playhead:
            self._shown_ts, self._frame = self._display.popleft()

    def _boxes_for_shown_frame(self) -> list:
        """Detections for the frame on screen (call with lock held)."""
        if self._shown_ts is None:
            return []
        found = []
        for ts, boxes in reversed(self._box_log):
            if ts <= self._shown_ts + 1e-6:
                found = boxes
                break
        # forget detections older than ~5 s behind the playhead
        while len(self._box_log) > 1 and self._box_log[1][0] <= self._shown_ts - 5:
            self._box_log.popleft()
        return found

    def _status_for_shown_frame(self):
        """Tracker state as it was at the frame on screen (call with lock held)."""
        if self._shown_ts is None:
            return None
        found = None
        for ts, snap in reversed(self._status_log):
            if ts <= self._shown_ts + 1e-6:
                found = snap
                break
        while len(self._status_log) > 1 and self._status_log[1][0] <= self._shown_ts - 5:
            self._status_log.popleft()
        return found

    def render(self, width: int, height: int, show_boxes: bool = True) -> np.ndarray | None:
        """Latest frame, letterboxed to (width, height), RGB, with overlays."""
        with self._frame_lock:
            self._advance_playback(time.time())
            frame = None if self._frame is None else self._frame.copy()
            boxes = list(self._boxes_for_shown_frame())
            shown_snap = self._status_for_shown_frame()
        width, height = max(width, 2), max(height, 2)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        snap = shown_snap or self.tracker.snapshot()

        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(width / fw, height / fh)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            if show_boxes:
                for (x1, y1, x2, y2), is_moving in boxes:
                    if is_moving is None:                       # too far away - ignored
                        color = (150, 150, 150)
                    elif is_moving or not self.ignore_stationary:
                        color = (0, 200, 0)                     # counts as passing
                    else:
                        color = (0, 165, 255)                   # stopped
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color,
                                  max(1, int(2 / max(scale, 0.25))))
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            x0, y0 = (width - nw) // 2, (height - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
        else:
            self._center_text(canvas, self.status)

        # status banner
        if frame is None or not snap.feed_ok:
            text, color = self.status if frame is not None else "", (90, 90, 90)
            if frame is not None and self.status == "Live":
                text = "Detecting..." if self.model_ready else "Loading YOLO model..."
        elif snap.in_event:
            text, color = f"BLOCKED  {fmt_duration(snap.current_event_seconds)}", (0, 0, 200)
        elif not snap.in_window:
            text, color = "Outside logging window", (90, 90, 90)
        elif snap.clear_seconds >= BANNER_CLEAR_S:
            text, color = f"No traffic for {snap.clear_seconds:.0f}s", (0, 140, 220)
        else:
            text, color = "Traffic moving", (0, 150, 0)
        if self.behind and text:
            text += "   (detector behind)"
        if text:
            cv2.rectangle(canvas, (0, 0), (width, 30), color, -1)
            cv2.putText(canvas, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _center_text(img: np.ndarray, text: str) -> None:
        h, w = img.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2, cv2.LINE_AA)


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
