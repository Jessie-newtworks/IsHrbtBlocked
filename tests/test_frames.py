"""Frame-by-frame detection tests.

They use a fake YOLO model (finds bright rectangles) so they run without
downloading weights or a GPU.  Run:  python -m pytest
"""

import sys
import threading
import time
import types
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from hrbt import error_log
from hrbt.event_tracker import EventTracker
from hrbt.stream_monitor import BoxHistory, StreamClock, StreamMonitor


@pytest.fixture(autouse=True)
def _tmp_error_log(tmp_path):
    error_log.set_log_path(tmp_path / "errors.log")


# ---------------------------------------------------------------- fake YOLO
class _Arr:
    def __init__(self, a):
        self.a = a

    def tolist(self):
        return self.a.tolist()


class _Boxes:
    def __init__(self, arr):
        self.xyxy = _Arr(arr)
        self._n = len(arr)

    def __len__(self):
        return self._n


class FakeYOLO:
    calls = 0

    def __init__(self, path):
        pass

    def predict(self, frame, **_):
        FakeYOLO.calls += 1
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n, _, stats, _ = cv2.connectedComponentsWithStats((g > 200).astype(np.uint8))
        arr = np.array([[x, y, x + w, y + h] for x, y, w, h, area in stats[1:] if area > 50],
                       float).reshape(-1, 4)
        return [types.SimpleNamespace(boxes=_Boxes(arr))]


@pytest.fixture
def fake_yolo(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    FakeYOLO.calls = 0
    return FakeYOLO


# ------------------------------------------------------------ helpers
FPS = 15
RNG = np.random.default_rng(0)


def road_frame():
    """Static grey road with a little sensor/compression noise."""
    base = np.full((360, 640, 3), 60, np.uint8)
    noise = RNG.integers(0, 4, base.shape, dtype=np.uint8)
    return cv2.add(base, noise)


def run_frames(monitor, frames_with_ts):
    """Push frames straight into the detector queue (bypassing the reader)."""
    monitor._stop.clear()
    t = threading.Thread(target=monitor._detector_loop, daemon=True)
    t.start()
    while not monitor.model_ready:
        time.sleep(0.01)
    for frame, ts in frames_with_ts:
        monitor._last_frame_wall = time.monotonic()
        monitor._push(frame, ts)
    deadline = time.time() + 20
    while monitor._queue and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    monitor._stop.set()
    with monitor._queue_cv:
        monitor._queue_cv.notify_all()
    t.join(2)


def make_monitor(tracker, motion_fraction=0.0005):
    return StreamMonitor(1, "S1", "unused", tracker, Path("fake.pt"), 0.35,
                         motion_fraction, True, 20)


T0 = time.mktime((2026, 10, 13, 12, 0, 0, 0, 0, -1))  # noon, inside 0600-2000


def stream(seconds, car_at=None, car_frames=3, stopped_car=False):
    """Yield (frame, ts) at 15 fps. car_at: list of seconds when a fast car
    crosses the view, visible for only `car_frames` frames."""
    car_at = car_at or []
    out = []
    for i in range(int(seconds * FPS)):
        ts = T0 + i / FPS
        f = road_frame()
        for start in car_at:
            k = i - int(start * FPS)
            if 0 <= k < car_frames:               # car zooms across: 200 px per frame
                x = 40 + 200 * k
                cv2.rectangle(f, (x, 150), (x + 90, 200), (255, 255, 255), -1)
        if stopped_car:
            cv2.rectangle(f, (500, 260), (590, 310), (255, 255, 255), -1)
        out.append((f, ts))
    return out


# --------------------------------------------------------------- tests
def test_fast_cars_in_three_frames_prevent_event(fake_yolo):
    # A car every 4 s, each visible for only 3 frames; threshold 5 s -> no event.
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr), stream(30, car_at=[1, 5, 9, 13, 17, 21, 25, 29]))
    assert tr.drain(flush=True) == {}


def test_two_frame_car_is_seen(fake_yolo):
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr), stream(12, car_at=[4], car_frames=2))
    # gaps: 0-4 s (<5) and 4.13-12 s (~7.9 s) -> exactly one event of ~7.9 s
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and 7 <= secs <= 8


def test_empty_exit_triggers_event_and_skips_yolo(fake_yolo):
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr), stream(20))
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and 19 <= secs <= 20     # whole 20 s of empty road
    assert fake_yolo.calls <= 3          # motion gate: static road needs almost no YOLO


def test_parked_car_does_not_hide_stoppage(fake_yolo):
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr, motion_fraction=0), stream(15, stopped_car=True))
    events, _ = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1


def test_stream_clock_smooths_hls_bursts_and_stalls():
    clock = StreamClock()
    # 2 s segments of 30 frames, each decoded in a burst. The first segment
    # arrives at t=100, the second on time at t=102, the third 10 s late.
    arrivals = [100.0, 102.0, 114.0]
    stamps = [clock.stamp((seg * 30 + i) * 1000 / FPS, arrivals[seg] + i * 0.001)
              for seg in range(3) for i in range(30)]
    gaps = np.diff(stamps)
    assert np.allclose(gaps, 1 / FPS, atol=1e-3)   # evenly spaced like the real video


def test_stream_clock_resyncs_on_restart():
    clock = StreamClock()
    a = clock.stamp(500_000, 100.0)
    b = clock.stamp(1_000, 200.0)        # PTS restarted
    assert b > a and abs(b - 200.0) < 1e-6


def test_gap_in_video_is_not_blocked_time(fake_yolo):
    # 3 s of video, then 20 s missing from the stream, then 3 s -> no event,
    # because the road was never *seen* empty for 5 s.
    tr = EventTracker(5, 360, 1200)
    frames = stream(3) + [(f, ts + 23) for f, ts in stream(3)]
    run_frames(make_monitor(tr), frames)
    assert tr.drain(flush=True) == {}


def test_box_history_references():
    h = BoxHistory((0.5, 1.0, 2.0))
    for t in (0.0, 0.4, 1.0, 1.6, 2.0):
        h.add(t, [t])
    assert h.references(1.9) is None           # nothing 2 s back yet
    assert h.references(2.1) == [[1.6], [1.0], [0.0]]
    h.add(2.5, [2.5])
    assert h.references(2.6) == [[2.0], [1.6], [0.4]]


def approaching_car(seconds, growth_per_s=0.15):
    """A car driving straight toward the camera: its box stays centred and grows
    ~15 %/s (a car ~100 m out at highway speed). Over 0.5 s it barely changes;
    over 2 s it clearly does."""
    out = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        f = road_frame()
        w = 50 * (1 + growth_per_s) ** t        # apparent size grows ever faster
        h = 35 * (1 + growth_per_s) ** t        # as the car gets closer
        cx, cy = 320, 120
        cv2.rectangle(f, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                      (255, 255, 255), -1)
        out.append((f, T0 + t))
    return out


def test_slow_approaching_car_counts_as_moving(fake_yolo):
    # Old 0.5 s check called this car "stopped", so a stoppage was logged
    # while traffic was moving. It must now count as passing -> no event.
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr, motion_fraction=0), approaching_car(12))
    assert tr.drain(flush=True) == {}


def test_cars_two_seconds_apart_count_as_moving(fake_yolo):
    # Steady traffic: same-size cars crossing the same spot every 2 s.
    # Comparing only with 2 s ago would make each car look parked.
    tr = EventTracker(5, 360, 1200)
    frames = []
    for i in range(20 * FPS):
        f = road_frame()
        k = i % 30                                  # a new car every 30 frames (2 s)
        if k < 24:                                  # crosses the view in 1.6 s
            x = 20 + 25 * k
            cv2.rectangle(f, (x, 150), (x + 60, 190), (255, 255, 255), -1)
        frames.append((f, T0 + i / FPS))
    run_frames(make_monitor(tr, motion_fraction=0), frames)
    assert tr.drain(flush=True) == {}


# ------------------------------------------------------- smooth live view
def _simulate_playback(segment_s=2.0, burst_segments=3, seconds=30, late=None):
    """HLS-style delivery: on connect, `burst_segments` segments arrive at once,
    then one segment every `segment_s` seconds (decoded in ~50 ms). The GUI
    refreshes 15 times a second. Returns the longest time the picture stood still
    after the first frame appeared. late: {segment_index: extra_delay_s}."""
    late = late or {}
    m = make_monitor(EventTracker(5, 0, 0))
    clock = StreamClock()
    frame = np.zeros((4, 4, 3), np.uint8)
    per_seg = int(segment_s * FPS)
    arrivals = []                                   # (arrival, pts_ms)
    for seg in range(int(seconds / segment_s)):
        t_seg = seg * 0.05 if seg < burst_segments else (seg - burst_segments + 1) * segment_s
        t_seg += late.get(seg, 0.0)
        for i in range(per_seg):
            arrivals.append((1000.0 + t_seg + i * 0.05 / per_seg, (seg * per_seg + i) * 1000 / FPS))
    arrivals.sort()

    shown_changes, k, now = [], 0, 1000.0
    while now < 1000.0 + seconds - 3:
        while k < len(arrivals) and arrivals[k][0] <= now:     # reader thread
            arrival, pts = arrivals[k]
            ts = clock.stamp(pts, arrival)
            m._update_lag(arrival, ts)
            m._display.append((ts, frame))
            k += 1
        before = m._shown_ts
        m._advance_playback(now)                                 # GUI refresh
        if m._shown_ts != before:
            shown_changes.append(now)
        now += 1 / 15
    return max(np.diff(shown_changes))


def test_live_view_plays_smoothly_despite_bursts():
    # Old behaviour (show newest decoded frame) froze for ~2 s per segment.
    assert _simulate_playback() < 0.2


def test_live_view_recovers_smoothly_after_a_late_segment():
    # one segment 3 s late: at most one pause, then smooth again
    assert _simulate_playback(late={6: 3.0}, seconds=40) < 3.5


# ------------------------------------------- repeated frames / banner / events.log
def test_repeated_frames_keep_moving_flag_and_events_are_logged(fake_yolo, tmp_path):
    # The camera sends 15 fps but only ~5 new pictures per second: each real
    # picture is repeated 3 times. A car driving through must show green
    # (moving) on the repeats too, not orange.
    tr = EventTracker(5, 0, 0)
    m = make_monitor(tr)
    frames = []
    for i in range(12 * FPS):
        real = i // 3                                  # new picture every 3rd frame
        f = road_frame() if i % 3 == 0 else frames[-1][0]
        if real < 20:                                  # car drives across for 4 s
            x = 20 + 30 * real
            f = f.copy()
            cv2.rectangle(f, (x, 150), (x + 60, 190), (255, 255, 255), -1)
        frames.append((f, T0 + i / FPS))
    run_frames(m, frames)
    car_entries = [boxes for ts, boxes in m._box_log if ts < T0 + 4 and boxes]
    assert car_entries and all(mv for boxes in car_entries[8:] for _, mv in boxes)
    # road empty for ~8 s after the car -> one event, written to events.log
    events, _ = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1
    log = (error_log.get_log_path().parent / "events.log").read_text()
    assert "event STARTED" in log


def test_banner_matches_frame_on_screen():
    m = make_monitor(EventTracker(5, 0, 0))
    from hrbt.event_tracker import TrackerSnapshot
    old = TrackerSnapshot(True, True, False, 0.0, 0.0)
    new = TrackerSnapshot(True, True, True, 9.0, 4.0)
    m._status_log.extend([(100.0, old), (105.0, new)])
    m._shown_ts = 102.0                 # picture is 3 s behind the detector
    assert m._status_for_shown_frame() is old
    m._shown_ts = 105.5
    assert m._status_for_shown_frame() is new


# ------------------------------- tonight's miss: wobbling false detection, empty tunnel
def wobbling_object(seconds, size, seed=1):
    """An empty tunnel with one static 'vehicle' (light / reflection) whose
    detected box wobbles +/-1 px in position and size every picture."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(seconds * FPS)):
        f = road_frame()
        dx, dy, dw = rng.integers(-1, 2, size=3)
        x, y = 300 + dx, 100 + dy
        cv2.rectangle(f, (x, y), (x + size[0] + dw, y + size[1] + dw), (255, 255, 255), -1)
        out.append((f, T0 + i / FPS))
    return out


def test_small_far_false_detection_does_not_hide_blockage(fake_yolo):
    # 20x20 px object far down the tunnel (below 10 % of picture height):
    # ignored, so the empty exit is logged as a stoppage.
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr, motion_fraction=0), wobbling_object(15, (20, 20)))
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and secs >= 8


def test_wobbling_static_box_counts_as_stopped(fake_yolo):
    # Bigger static object (50x40) with box wobble: must count as stopped,
    # not as a passing car, so the blockage is still logged.
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr, motion_fraction=0), wobbling_object(15, (50, 40)))
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and secs >= 7


def test_small_wobbling_static_box_counts_as_stopped(fake_yolo):
    # just above the size cut-off, wobbling in position AND size
    tr = EventTracker(5, 360, 1200)
    run_frames(make_monitor(tr, motion_fraction=0), wobbling_object(15, (26, 24)))
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and secs >= 7


def test_small_car_seen_in_one_picture_counts(fake_yolo):
    # camera sends ~1.6 real pictures/s: a smallish car (30x20) caught in a
    # single picture must still reset the timer
    tr = EventTracker(5, 360, 1200)
    frames = []
    for i in range(12 * FPS):
        real = i // 9                                   # new picture every 9 frames
        f = road_frame() if i % 9 == 0 else frames[-1][0]
        if real == 8:                                   # car in one picture at ~4.8 s
            f = f.copy()
            cv2.rectangle(f, (100, 60), (130, 80), (255, 255, 255), -1)
        frames.append((f, T0 + i / FPS))
    run_frames(make_monitor(tr), frames)
    # without the car: one 12 s event; with it: only the ~7 s after it is blocked
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and secs <= 7.5        # would be ~12 s if the car were missed
